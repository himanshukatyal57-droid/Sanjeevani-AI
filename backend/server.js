
const express = require("express");
const dotenv = require("dotenv");
const path = require("path");
const analysisRoutes = require("./routes/analysis.routes");

// Load environment variables
dotenv.config();

const app = express();

// Middleware
app.use(express.json());

// API Routes
app.use("/api/analysis", analysisRoutes);

// Serve frontend files
app.use(express.static(path.join(__dirname, "../Frontend")));

// Health check API
app.get("/api/health", (req, res) => {
    res.status(200).json({
        success: true,
        message: "Sanjeevani AI backend is running!",
        status: "healthy"
    });
});

// Server configuration
const PORT = process.env.PORT || 5000;

// Start server
app.listen(PORT, () => {
    console.log(`Sanjeevani AI server running on port ${PORT}`);
});

