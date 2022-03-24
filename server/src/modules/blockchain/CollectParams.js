const fs = require("fs");
const md5 = require("md5");

module.exports = (params) => {
    try {
        collectparams = {
            ModelID: params[0],
            ClientKey: params[1],
            ClientEpochs: params[2],
            UpdateIteration: params[3],
        };

        let collection_str = fs.readFileSync(`./models/${params[0]}U.json`);
        let collected_params = JSON.parse(collection_str);
        iteration = collected_params.Iteration
        if (iteration == -1)
            collected_params.Iteration = params[3];
        console.log("storing client params of client", collectparams.ClientKey);
        fs.writeFileSync(`./all_params/${collectparams.ClientKey}.json`, JSON.stringify(collectparams));
        collected_params.AllParams[params[1]]= params[1];
        console.log("Client Params collected", collectparams.ClientKey);
        console.log("collected all local params: ", Object.values(collected_params.AllParams));
        fs.writeFileSync(`./models/${collected_params.ModelID}U.json`, JSON.stringify(collected_params));
        console.log("Created File for collected params", collected_params.ModelID);

        if (collected_params.AllParams.length== collected_params.NClients)
            collected_params.Lock = true;
            fs.writeFileSync(`./models/${collected_params.ModelID}U.json`, JSON.stringify(collected_params));
            console.log("Update params lock updated")
        return collectparams;
    } catch (error) {
        console.error(error);
        return null;
    }
};
