const express = require("express");
const app = express();
const cors = require("cors");
const dotenv = require("dotenv").config();
const twilio = require("twilio");
const PORT = process.env.PORT || 9000;
app.use(express.json());
app.use(
  cors({
    origin: "*",
  })
);

app.post("/api/emergency-notification", (req, res) => {
  const { name, phone, location } = req.body;
  const emergencyNotification = {
    name,
    phone,
    location,
  };
  console.log(emergencyNotification);

  const accountSid = process.env.TWILIO_ACCOUNT_SID;
  const authToken = process.env.TWILIO_AUTH_TOKEN;
  const client = require("twilio")(accountSid, authToken);

  client.messages
    .create({
      body: `Emergency Alert! Abnormal brainwave activity detected for ${name}. Location: ${location}. Please take immediate action.`,
      to: phone,
      from: process.env.TWILIO_PHONE_NUMBER,
    })
    .then((message) => console.log(`Message sent: ${message.sid}`))
    .catch((error) => console.error("Twilio error:", error));
});

app.listen(PORT, () => {
  console.log("Server is running on port", PORT);
});
