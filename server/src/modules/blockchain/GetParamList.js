const fs = require("fs");

module.exports = (ModelID) => {
    try{
        let all_params_str = fs.readFileSync(`./model/${ModelID}all_params.json`);
        let all_params = JSON.parse(all_params_str);
        console.log("Get all params list", all_params.ModelID);
        return all_params;
    }
    catch (error){
        console.error(error);
        return null;
    }
};