const fs = require("fs");

module.exports = (key) => {
    try {
        let value = fs.unlinkSync(`./models/shards/${key}.shard`);
        console.log("Deleted Client Model Shard", key);
        return value;
    } catch (error) {
        console.error(error);
        return null;
    }
};
