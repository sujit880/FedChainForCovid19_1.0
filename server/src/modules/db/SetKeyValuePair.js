const fs = require("fs");
const md5 = require("md5");

module.exports = (value) => {
    try {
        const key = md5(value);
        fs.writeFileSync(`./models/shards/${key}.shard`, value);
        console.log("Model Shard Set", key);
        return key;
    } catch (error) {
        console.error(error);
        return null;
    }
};
