const fs = require("fs");
const Blockchain = require("../blockchain");

module.exports = (params) => {
    try {
        console.log("got getLock request");
        let ModelLock = Blockchain.GetLock(params);

        console.log("Got Model Lock data", params[0], ModelLock);
        return ModelLock;
    } catch (error) {
        console.error(error);
        return null;
    }
};