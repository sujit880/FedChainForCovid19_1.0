const fs = require("fs");
const md5 = require("md5");

module.exports = (params) => {
    try {
        const key = params[0];
        console.log("store client params with key:->", key)
        fs.writeFileSync(`./all_params/${key}.shard`, params[1]);
        console.log("Model Shard Set", key);
        return key;
    } catch (error) {
        console.error(error);
        return null;
    }
};