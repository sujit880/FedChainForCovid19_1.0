const fs = require("fs");
const ShardDB = require("../db");

module.exports = (params) => {
    try {
        model = {
            ModelID: params[0],
            ModelParams: params[1],
            LearningRate: params[2],
            NPush: 50,
            Iteration: -1,
            ModelUpdateCount: -1,
            ModelReadLock: false,
            ModelUpdateLock: true,
        };

        fs.writeFileSync(`./models/${model.ModelID}.json`, JSON.stringify(model));
        console.log("Model Params Set", model.ModelID);
        // client_params_value = ShardDB.GetKeyValuePair(params[1]);
        // client_params_key = ShardDB.SetClientParamsPair([client_key , client_params_value]);
        collectedparams = {
            ModelID: params[0],            
            Iteration: -1,
            NClients:1,
            AllParams: {},
            AllClients: [],
            Lock: false,
            Scores: {},

        };
        // collectedparams.AllClients.push("hello")
        fs.writeFileSync(`./models/${collectedparams.ModelID}U.json`, JSON.stringify(collectedparams));
        console.log("Created File for collected params", collectedparams.ModelID);
        return model;
    } catch (error) {
        console.error(error);
        return null;
    }
};
