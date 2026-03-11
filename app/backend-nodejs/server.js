require("dotenv").config();  

const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const apiRoutes = require('./routes/api.routes');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api', apiRoutes);

// Health check
app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        service: 'Node.js API Backend',
        timestamp: new Date().toISOString()
    });
});

// Error handling
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ 
        success: false, 
        error: 'Internal Server Error',
        message: err.message 
    });
});

app.listen(PORT, () => {
    console.log('='.repeat(50));
    console.log(`🚀 NODE.JS API BACKEND STARTED`);
    console.log(`📡 Server running on port ${PORT}`);
    console.log(`🔗 API Endpoint: http://localhost:${PORT}/api`);
    console.log('='.repeat(50));
});