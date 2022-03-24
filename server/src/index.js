const express = require("express");
const cors = require("cors");
require("dotenv").config();

const app = express();

app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ limit: "50mb" }));

// JSON Middleware
app.use(express.json());

// CORS Middleware
app.use(cors());
app.use(function (req, res, next) {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE");
    res.header("Access-Control-Allow-Headers", "Content-Type");
    next();
});

// Load API Routes
require("./routes")(app);

// Start Listening
app.listen(5500, '0.0.0.0', () => console.log("Listening on port 5500."));
