const nodemailer = require('nodemailer');
const config = require('../config/config');

const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: config.email.user,
        pass: config.email.password
    }
});

exports.sendAnalysisEmail = async (result, breed) => {
    try {
        const bcs = result.bcs_score || 'N/A';
        const milk = result.milk_production?.estimated_avg || 'N/A';

        const mailOptions = {
            from: config.email.user,
            to: config.email.adminEmail,
            subject: `Analysis: BCS ${bcs} | Milk ~${milk} L/day`,
            html: `
                
                
                    
                        New Cow Analysis
                        
                            ${bcs}
                            BCS Score
                        
                        
                            ${milk} L/day
                            Estimated Production
                        
                        Breed: ${breed}
                        Timestamp: ${new Date().toLocaleString()}
                    
                
                
            `,
            attachments: []
        };

        // Image attachments have been disabled per user request
        /* 
        if (result.side_uploaded_image) {
            const sideImageData = result.side_uploaded_image.split(',')[1];
            mailOptions.attachments.push({
                filename: 'side_view.jpg',
                content: sideImageData,
                encoding: 'base64'
            });
        }

        if (result.rear_uploaded_image) {
            const rearImageData = result.rear_uploaded_image.split(',')[1];
            mailOptions.attachments.push({
                filename: 'rear_view.jpg',
                content: rearImageData,
                encoding: 'base64'
            });
        }
        */

        await transporter.sendMail(mailOptions);
        console.log(`✅ Email sent to ${config.email.adminEmail}`);
        return true;
    } catch (error) {
        console.error('❌ Email failed:', error);
        throw error;
    }
};