const fs = require("fs");
const Blockchain = require("../blockchain");
const ShardDB = require("../db");

module.exports = (params) => {
    try {
        params[1] = ShardDB.SetKeyValuePair(params[1]);

        const model = Blockchain.SetModel(params);
        console.log("Retriving model params: ->", model.ModelParams)
        model.ModelParams = ShardDB.GetKeyValuePair(model.ModelParams);
        console.log("Assembled Model Params Set", model.ModelID);
        return model;
    } catch (error) {
        console.error(error);
        return null;
    }
};
