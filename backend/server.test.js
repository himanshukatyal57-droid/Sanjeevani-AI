const { test, before, after } = require("node:test");
const assert = require("node:assert");
const app = require("./server");

let server;
let url;

before(async () => {
    await new Promise(resolve => {
        server = app.listen(0, "127.0.0.1", resolve);
    });
    url = `http://127.0.0.1:${server.address().port}`;
});

after(() => server.close());

function post(body, headers = { "Content-Type": "application/json" }) {
    return fetch(`${url}/api/analysis`, { method: "POST", headers, body });
}

// Every failure must be JSON, with no stack trace or file paths
async function assertCleanJsonError(response, status) {
    const text = await response.text();
    assert.strictEqual(response.status, status);
    assert.match(response.headers.get("content-type"), /json/);
    assert.strictEqual(JSON.parse(text).success, false);
    assert.doesNotMatch(text, /node_modules|SyntaxError|\sat\s/);
}

test("malformed JSON is a clean 400", async () => {
    await assertCleanJsonError(await post('{"symptoms": '), 400);
});

test("JSON null is a clean 400", async () => {
    await assertCleanJsonError(await post("null"), 400);
});

test("an oversized body is a clean 413", async () => {
    await assertCleanJsonError(await post(JSON.stringify({ symptoms: "a".repeat(120000) })), 413);
});

test("a body that isn't JSON is a 400, not 'AI unavailable'", async () => {
    await assertCleanJsonError(await post("I have a cough", { "Content-Type": "text/plain" }), 400);
});

test("unknown API routes are JSON", async () => {
    await assertCleanJsonError(await fetch(`${url}/api/nope`), 404);
});

test("the health route and the website still work", async () => {
    assert.strictEqual((await fetch(`${url}/api/health`)).status, 200);
    assert.match(await (await fetch(url)).text(), /Sanjeevani/);
});
