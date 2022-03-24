const util = require("util");

// Set Connection String
const connectionString = `redis://${process.env.REDIS_URI}`;

// Promisify
client.get = util.promisify(client.get);
const redis = require("redis");
const client = redis.createClient(connectionString);

module.exports = client;
