const fs = require("fs");
const md5 = require("md5");

module.exports = (params) => {
    try{
        
        client_key = params[2]+params[1];
        const key = md5(client_key);
        let model_str = fs.readFileSync(`./models/${params[0]}.json`);
        let model = JSON.parse(model_str);
        
        console.log("Model_params_get", model.ModelID);

        let collection_str = fs.readFileSync(`./models/${model.ModelID}U.json`);
        let collected_params = JSON.parse(collection_str);
        iteration = collected_params.Iteration;
        // collected_params.AllClients.push("hello")
        console.log("All clients", collected_params.AllClients);
        if (!collected_params.AllClients.includes(key)){
            collected_params.AllClients.push(key);
            collected_params.NClients = collected_params.AllClients.length;
            
        }
        if ( collected_params.Scores[key] === undefined) {
            collected_params.Scores[key] = 200;
            console.log("New score is given to client key", key);
        }
        console.log("ALL clients scores: ", Object.keys(collected_params.Scores), " -> ", Object.values(collected_params.Scores));
        fs.writeFileSync(`./models/${collected_params.ModelID}U.json`, JSON.stringify(collected_params));
        console.log("Updated collected params", collected_params.ModelID);
        return model;
    }
    catch (error){
        console.error(error);
        return null;
    }
};

