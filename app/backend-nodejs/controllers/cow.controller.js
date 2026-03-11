const mlService = require('../services/ml.service');
const emailService = require('../services/email.service');

exports.analyzeCow = async (req, res) => {
    try {
        const { breed, lactation_stage, parity, camera_distance } = req.body;
        const rearImage = req.files['rear_image'] ? req.files['rear_image'][0] : null;
        const sideBodyImage = req.files['side_body_image'] ? req.files['side_body_image'][0] : null;

        if (!rearImage) {
            return res.status(400).json({
                success: false,
                error: 'Rear image is required'
            });
        }

        // Process images with Python ML service
        const result = await mlService.processImages({
            rearImage,
            sideBodyImage,
            breed,
            lactation_stage,
            parity,
            camera_distance
        });

        if (!result.success) {
            return res.json(result);
        }

        /* Email notifications disabled per user request
        try {
            await emailService.sendAnalysisEmail(result, breed);
        } catch (emailError) {
            console.error('Email send failed:', emailError);
        }
        */

        return res.json(result);

    } catch (error) {
        console.error('Analysis error:', error);
        return res.status(500).json({
            success: false,
            error: 'Analysis Failed',
            message: error.message
        });
    }
};