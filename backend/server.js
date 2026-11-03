
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

// Unknown API routes
app.use("/api", (req, res) => {
    res.status(404).json({
        success: false,
        message: "API route not found."
    });
});

// Replaces Express's default error page, which shows a stack trace with file paths.
// The four arguments are how Express recognizes an error handler.
app.use((error, req, res, next) => {
    const status = error.status || 500;

    if (status >= 500) {
        console.error("Server error:", error);
    }

    res.status(status).json({
        success: false,
        message: status === 413 ? "That request is too large." : "The request could not be processed."
    });
});

// Server configuration
const PORT = process.env.PORT || 5000;

// Start server, unless a test is importing the app
if (require.main === module) {
    // Express passes a failure such as "port in use" to this callback instead of throwing
    app.listen(PORT, (error) => {
        if (error) {
            console.error(`Could not start on port ${PORT}: ${error.message}`);
            console.error("An old copy may still be running. See \"Stop the servers\" in README.md.");
            process.exit(1);
        }

        console.log(`Sanjeevani AI server running on port ${PORT}`);
    });
}

module.exports = app;

