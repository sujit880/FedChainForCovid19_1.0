const fs = require("fs");
const ShardDB = require("../db");

module.exports = (params) => {
    try {
        model = params[0];
        model.Iteration += 1; 
        model.ModelUpdateCount +=1;
        model.ModelReadLock = false;

        let collection_str = fs.readFileSync(`./model/${model.ModelID+'U'}.json`);
        let collected_params = JSON.parse(collection_str);
        let remove_clients_list = []
        for(let i=o; i<collected_params.AllClients.length;i++){
            if (collected_params.AllParams.include(collected_params.AllClients[i])){
                ShardDB.DeleteClientParamsPair(collected_params.AllClients[i]);
            }
            else{
                remove_clients_list.push(collected_params.AllClients[i]);
            }
        }

        for (let i=0; i<remove_clients_list.length; i++){
            const index = collected_params.AllClients.indexOf(remove_clients_list[i]);
            if (index > -1) {
                collected_params.AllClients.splice(index, 1); // 2nd parameter means remove one item only
            }
        }

        collected_params.NClients = collected_params.AllClients.length;
        // while(collected_params.AllParams.length>0){
        //     ShardDB.DeleteClientParamsPair(collected_params.AllParams.pop());
        // }

        
        fs.writeFileSync(`./models/${model.ModelID}.json`, JSON.stringify(model));
        console.log("Model Params Set", model.ModelID);

        fs.writeFileSync(`./models/${collectedparams.ModelID+'U'}.json`, JSON.stringify(collectedparams));
        console.log("Created File for collected params", collectedparams.ModelID);
        return model;
    } catch (error) {
        console.error(error);
        return null;
    }
};
