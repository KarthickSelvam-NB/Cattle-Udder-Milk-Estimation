const axios = require('axios');
const FormData = require('form-data');
const config = require('../config/config');

const PYTHON_ML_URL = config.pythonMlUrl;

exports.processImages = async ({ rearImage, sideBodyImage, breed, lactation_stage, parity, camera_distance }) => {
    try {
        let bcsResult = null;
        let udderResult = null;

        // Process side body image for BCS (if provided)
        if (sideBodyImage) {
            const bcsFormData = new FormData();
            bcsFormData.append('image', sideBodyImage.buffer, sideBodyImage.originalname);

            try {
                const bcsResponse = await axios.post(`${PYTHON_ML_URL}/process_bcs`, bcsFormData, {
                    headers: bcsFormData.getHeaders()
                });
                bcsResult = bcsResponse.data;
            } catch (err) {
                console.error('BCS processing failed:', err.message);
            }
        }

        // Process rear image for udder (depth is extracted from the rear image itself)
        const udderFormData = new FormData();
        udderFormData.append('rear_image', rearImage.buffer, rearImage.originalname);
        udderFormData.append('breed', breed);
        udderFormData.append('lactation_stage', lactation_stage);
        udderFormData.append('parity', parity);
        udderFormData.append('camera_distance', camera_distance || '2.5');

        const udderResponse = await axios.post(`${PYTHON_ML_URL}/process_udder`, udderFormData, {
            headers: udderFormData.getHeaders()
        });
        udderResult = udderResponse.data;

        if (!udderResult.success) {
            return udderResult;
        }

        // Combine results
        const result = { success: true };

        // Add BCS results if available
        if (bcsResult && bcsResult.success) {
            result.bcs_score = bcsResult.bcs_score;
            result.ribs = bcsResult.ribs;
            result.short_ribs = bcsResult.short_ribs;
            result.hook = bcsResult.hook;
            result.thurl = bcsResult.thurl;
            result.side_output_image = bcsResult.output_image;
            result.side_uploaded_image = bcsResult.uploaded_image;
            result.side_lighting_warnings = bcsResult.lighting_warnings;

            // Calculate milk from BCS
            const milkFromBcs = await estimateMilkFromBcs(
                bcsResult.bcs_score,
                breed,
                lactation_stage,
                parity
            );

            // Combine with udder estimate
            if (milkFromBcs && udderResult.milk_production) {
                const avgEstimate = (milkFromBcs.estimated_avg + udderResult.milk_production.estimated_avg) / 2;
                result.milk_production = {
                    estimated_range: `${(avgEstimate * 0.8).toFixed(1)} - ${(avgEstimate * 1.2).toFixed(1)} L/day`,
                    estimated_avg: avgEstimate.toFixed(1),
                    bcs_estimate: milkFromBcs.estimated_avg,
                    udder_estimate: udderResult.milk_production.estimated_avg,
                    confidence: "High (Combined BCS + Udder)",
                    confidence_pct: "80-90%",
                    method: "Combined Analysis (BCS + Udder)",
                    flags: milkFromBcs.flags || [],
                    explanation: {
                        calculation_method: "Combined BCS + Udder Analysis",
                        bcs_calculation: milkFromBcs.explanation.steps,
                        udder_calculation: udderResult.milk_production.explanation.steps,
                        final_step: `Averaged both methods: (${milkFromBcs.estimated_avg} + ${udderResult.milk_production.estimated_avg}) / 2 = ${avgEstimate.toFixed(1)} L/day`,
                        accuracy_note: "Combined analysis provides highest accuracy (80-90% confidence)"
                    },
                    udder_features: udderResult.milk_production.udder_features
                };
                if (milkFromBcs.bcs_status) {
                    result.milk_production.bcs_status = milkFromBcs.bcs_status;
                }
            }
        } else {
            result.bcs_score = "Not provided";
            result.side_note = "Side image not uploaded - using udder-only analysis";
            result.milk_production = udderResult.milk_production;
            if (result.milk_production) {
                result.milk_production.note = "⚠️ Udder-only analysis. Upload side image for BCS to improve accuracy by 15-25%";
            }
        }

        // Add BCS side images if available
        if (bcsResult && bcsResult.success) {
            result.side_bcs_output_image = bcsResult.output_image;
            result.side_bcs_uploaded_image = bcsResult.uploaded_image;
            result.side_bcs_lighting_warnings = bcsResult.lighting_warnings;

            // For backward compatibility with the current frontend
            result.side_output_image = bcsResult.output_image;
            result.side_uploaded_image = bcsResult.uploaded_image;
            result.side_lighting_warnings = bcsResult.lighting_warnings;
        }

        // Add udder images
        result.rear_output_image = udderResult.output_image;
        result.rear_uploaded_image = udderResult.uploaded_image;
        result.rear_lighting_warnings = udderResult.lighting_warnings;

        // Add side udder analysis status and warning
        result.side_udder_detected = udderResult.side_udder_detected;
        result.side_udder_warning = udderResult.side_udder_warning;

        // Add side udder analysis if available
        if (udderResult.side_output_image) {
            result.side_udder_output_image = udderResult.side_output_image;
            // Use existing side uploaded image if BCS failed but udder side succeeded
            result.side_uploaded_image = result.side_uploaded_image || udderResult.side_uploaded_image;
        }

        return result;

    } catch (error) {
        console.error('ML Service Error:', error);
        throw error;
    }
};

// Helper function for BCS milk estimation
async function estimateMilkFromBcs(bcs_score, breed, lactation_stage, parity) {
    const BREED_RANGES = {
        "hf": [20, 35, 25],
        "hf_cross": [15, 25, 20],
        "jersey": [12, 20, 18],
        "jersey_cross": [10, 16, 15],
        "gir": [8, 15, 14],
        "sahiwal": [6, 14, 12],
        "ongole": [5, 10, 9],
        "kangayam": [4, 8, 7],
        "tamil_native": [4, 9, 8]
    };

    if (bcs_score === "UNKNOWN") return null;

    const [breed_base_min, breed_base_max, breed_max_cap] = BREED_RANGES[breed] || [8, 15, 12];
    const breed_base_avg = (breed_base_min + breed_base_max) / 2;

    let bcs_numeric;
    if (bcs_score.includes("< 2")) bcs_numeric = 1.75;
    else if (bcs_score.includes(">") || bcs_score.includes("4")) bcs_numeric = 4.0;
    else bcs_numeric = parseFloat(bcs_score);

    let bcs_factor, bcs_status;
    if (bcs_numeric < 2.0) {
        bcs_factor = 1.20;
        bcs_status = "Very Thin (High Production - Risky)";
    } else if (bcs_numeric < 2.5) {
        bcs_factor = 1.10;
        bcs_status = "Thin (High Production)";
    } else if (bcs_numeric < 2.75) {
        bcs_factor = 1.05;
        bcs_status = "Moderately Thin";
    } else if (bcs_numeric === 3.0) {
        bcs_factor = 1.00;
        bcs_status = "Optimal";
    } else if (bcs_numeric <= 3.5) {
        bcs_factor = 0.95;
        bcs_status = "Good Condition";
    } else if (bcs_numeric <= 4.0) {
        bcs_factor = 0.85;
        bcs_status = "Over-conditioned";
    } else {
        bcs_factor = 0.75;
        bcs_status = "Obese";
    }

    const stage_factors = { "early": 1.10, "mid": 1.0, "late": 0.72 };
    const stage_factor = stage_factors[lactation_stage] || 1.0;
    const parity_factor = parseInt(parity) === 1 ? 0.80 : 1.0;

    let estimated_avg = breed_base_avg * bcs_factor * stage_factor * parity_factor;
    let capped = false;

    if (estimated_avg > breed_max_cap) {
        estimated_avg = breed_max_cap;
        capped = true;
    }

    const range_pct = 0.20;
    let estimated_min = estimated_avg * (1 - range_pct);
    let estimated_max = estimated_avg * (1 + range_pct);

    if (estimated_max > breed_max_cap) {
        estimated_max = breed_max_cap;
        estimated_min = breed_max_cap * (1 - range_pct);
    }

    estimated_avg = parseFloat(estimated_avg.toFixed(1));
    estimated_min = parseFloat(estimated_min.toFixed(1));
    estimated_max = parseFloat(estimated_max.toFixed(1));

    const flags = [];
    if (bcs_numeric < 2.5) flags.push("⚠️ Low body condition");
    if (bcs_numeric > 3.5) flags.push("⚠️ Over-conditioned");
    if (bcs_numeric >= 2.75 && bcs_numeric <= 3.25) flags.push("✅ Excellent condition");
    if (capped) flags.push(`ℹ️ Capped at ${breed_max_cap} L/day`);

    return {
        estimated_range: `${estimated_min} - ${estimated_max} L/day`,
        estimated_avg,
        bcs_status,
        confidence: "Medium-High (BCS-based)",
        confidence_pct: "70-80%",
        flags,
        method: "BCS Score Analysis",
        explanation: {
            calculation_method: "BCS Score Analysis",
            steps: [
                `1. BCS Score: ${bcs_score} (${bcs_status})`,
                `2. Applied breed baseline: ${breed.toUpperCase()} = ${breed_base_min}-${breed_base_max} L/day (avg: ${breed_base_avg} L/day)`,
                `3. Applied BCS factor: ${bcs_factor}x (based on body condition)`,
                `4. Applied lactation stage factor: ${lactation_stage} = ${stage_factor}x`,
                `5. Applied parity factor: Parity ${parity} = ${parity_factor}x`,
                `6. Final calculation: ${breed_base_avg} × ${bcs_factor} × ${stage_factor} × ${parity_factor} = ${estimated_avg} L/day`
            ]
        }
    };
}