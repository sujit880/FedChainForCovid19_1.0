const fs = require("fs");
const Blockchain = require("../blockchain");
const SharDb = require("../db");

module.exports = (modelID) => {
    try{
        let bModel = Blockchain.GetModel(modelID);
        
        let all_params = Blockchain.GetClientParams(modelID);
        if (all_params !==null){
            bModel.ModelReadLock = true;
        }
        fs.writeFileSync(`./models/${bModel.ModelID}.json`, JSON.stringify(bModel));
        console.log("Model Lock status update..", bModel.ModelID);
        return all_params;
    }catch (error) {
        console.error(error);
        return null;
    }
};