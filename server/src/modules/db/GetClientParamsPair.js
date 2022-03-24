const fs = require("fs");

module.exports = (key) => {
    try {
        let value = fs.readFileSync(`./models/shards/${key}.shard`, "utf8");
        console.log("Get Client Model Shard", key);
        return value;
    } catch (error) {
        console.error(error);
        return null;
    }
};