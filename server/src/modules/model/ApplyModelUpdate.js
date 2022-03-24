const fs = require("fs");
const Blockchain = require("../blockchain");
const ShardDB = require("../db");

module.exports = (params) => {
    try {
        let bModel = Blockchain.GetModel([params[0]]);
        const modelHistory = parseInt(process.env.NUM_MODEL_HISTORY);

        // Delete Old Param Shards
        ShardDB.DeleteKeyValuePair(bModel.ModelParams); // Delete old global params

        //
        bModel.ModelParams = ShardDB.SetKeyValuePair(params[1]);

        // Set Model History Length
        params.push(process.env.NUM_MODEL_HISTORY);

        const model = Blockchain.ApplyModelUpdate(bModel);
        console.log("Assembled Model Update Set", params[0]);
        return model;
    } catch (error) {
        console.error(error);
        return null;
    }
};
