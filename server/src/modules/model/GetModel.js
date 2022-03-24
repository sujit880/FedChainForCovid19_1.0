const fs = require("fs");
const Blockchain = require("../blockchain");
const SharDb = require("../db");
const md5 = require("md5");

module.exports = (params) => {
    try{
        console.log("got fetch param request");
        client_key = params[1]+params[2];
        const key = md5(client_key);
        params.push(key);
        let bModel = Blockchain.GetModel(params);
        bModel.ModelParams = SharDb.GetKeyValuePair(bModel.ModelParams);
        
        console.log("Get Assembled Model Params", bModel.ModelID);
        return bModel;
    }catch (error) {
        console.error(error);
        return null;
    }
};