const express = require("express")
const app = express()
const cors = require("cors")
const dotenv = require("dotenv").config()
const PORT = process.env.PORT || 8000

app.use(cors({
    origin:"*"
}))

app.use("/api/user", userRoutes)

app.listen(PORT, ()=>{
    console.log("Server is running on port", PORT)
})