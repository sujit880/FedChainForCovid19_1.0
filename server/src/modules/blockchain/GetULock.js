const fs = require("fs");

module.exports = (ModelID) => {
    try {
        let umodstr = fs.readFileSync(`./models/${ModelID}U.json`);
        let umodel = JSON.parse(umodstr);
        console.log("Model Lock Status Get", umodel.ModelID);
        console.log("update Lock->", umodel.Lock);
        return umodel.Lock;
    } catch (error) {
        console.log("Error occured");
        console.error(error);
        return null;
    }
};