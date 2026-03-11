module.exports = {
    port: process.env.PORT || 3000,
    pythonMlUrl: process.env.PYTHON_ML_URL || 'http://localhost:5000',
    email: {
        user: process.env.GMAIL_USER,
        password: process.env.GMAIL_APP_PASSWORD,
        adminEmail: process.env.ADMIN_EMAIL
    }
};