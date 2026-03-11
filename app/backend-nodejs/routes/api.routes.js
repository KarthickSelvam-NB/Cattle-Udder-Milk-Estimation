const express = require('express');
const multer = require('multer');
const cowController = require('../controllers/cow.controller');

const router = express.Router();

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Routes
router.post('/predict',
    upload.fields([
        { name: 'rear_image', maxCount: 1 },
        { name: 'side_body_image', maxCount: 1 }
    ]),
    cowController.analyzeCow
);

module.exports = router;