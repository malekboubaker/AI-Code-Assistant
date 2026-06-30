"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.ApiClient = void 0;
const http = __importStar(require("node:http"));
const https = __importStar(require("node:https"));
class ApiClient {
    endpoint;
    timeoutMs;
    apiBaseUrl;
    constructor(endpoint, timeoutMs) {
        this.endpoint = endpoint;
        this.timeoutMs = timeoutMs;
        assertLocalhostEndpoint(endpoint);
        this.apiBaseUrl = apiBaseFromGenerateEndpoint(endpoint);
    }
    async generate(request) {
        const response = await postJson(this.endpoint, request, this.timeoutMs);
        return validateGenerateResponse(response);
    }
    streamGenerate(request, onEvent) {
        const url = new URL(`${this.apiBaseUrl}/generate/stream`);
        const data = Buffer.from(JSON.stringify(request), 'utf8');
        const transport = url.protocol === 'https:' ? https : http;
        return new Promise((resolve, reject) => {
            const req = transport.request({
                method: 'POST',
                hostname: url.hostname,
                port: url.port,
                path: `${url.pathname}${url.search}`,
                headers: {
                    'Content-Type': 'application/json',
                    'Content-Length': data.length,
                },
                timeout: this.timeoutMs,
            }, (res) => {
                if (!res.statusCode || res.statusCode >= 400) {
                    let body = '';
                    res.on('data', (chunk) => { body += chunk.toString('utf8'); });
                    res.on('end', () => reject(new Error(`Backend streaming error (${res.statusCode}): ${body}`)));
                    return;
                }
                let buffer = '';
                res.on('data', (chunk) => {
                    buffer += chunk.toString('utf8');
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';
                    for (const line of lines) {
                        if (!line.trim())
                            continue;
                        try {
                            const event = JSON.parse(line);
                            if (event.type === 'result') {
                                try {
                                    resolve(validateGenerateResponse(event.payload));
                                }
                                catch (valErr) {
                                    reject(valErr);
                                }
                            }
                            else {
                                onEvent(event);
                            }
                        }
                        catch (e) {
                            // Ignore JSON parse errors for incomplete lines, they are handled by the buffer.
                        }
                    }
                });
                res.on('end', () => {
                    if (buffer.trim()) {
                        try {
                            const event = JSON.parse(buffer);
                            if (event.type === 'result') {
                                try {
                                    resolve(validateGenerateResponse(event.payload));
                                }
                                catch (valErr) {
                                    reject(valErr);
                                }
                            }
                            else
                                onEvent(event);
                        }
                        catch (e) {
                            reject(new Error("Stream ended with incomplete or invalid JSON: " + buffer));
                        }
                    }
                    else {
                        reject(new Error("Stream ended without a final result payload."));
                    }
                });
            });
            req.on('error', reject);
            req.on('timeout', () => {
                req.destroy();
                reject(new Error(`Backend request timed out after ${this.timeoutMs} ms.`));
            });
            req.write(data);
            req.end();
        });
    }
    async cancelRequest(responseId) {
        await postJson(`${this.apiBaseUrl}/cancel/${encodeURIComponent(responseId)}`, {}, this.timeoutMs);
    }
    async getRagStatus(projectPath) {
        const url = `${this.apiBaseUrl}/rag/status?project_path=${encodeURIComponent(projectPath)}`;
        const response = await getJson(url, this.timeoutMs);
        return validateRagStatusResponse(response);
    }
    async indexProject(projectPath, mode) {
        const response = await postJson(`${this.apiBaseUrl}/rag/index`, { project_path: projectPath, mode }, this.timeoutMs);
        return validateRagIndexResponse(response);
    }
    async resetProjectIndex(projectPath) {
        const response = await postJson(`${this.apiBaseUrl}/rag/reset`, { project_path: projectPath }, this.timeoutMs);
        return validateRagResetResponse(response);
    }
}
exports.ApiClient = ApiClient;
function assertLocalhostEndpoint(endpoint) {
    const url = new URL(endpoint);
    const allowedHosts = new Set(['localhost', '127.0.0.1', '::1']);
    if (!allowedHosts.has(url.hostname)) {
        throw new Error(`Refusing non-local backend URL: ${endpoint}`);
    }
    if (url.protocol !== 'http:') {
        throw new Error(`Only local http:// backend URLs are allowed: ${endpoint}`);
    }
}
function postJson(endpoint, payload, timeoutMs) {
    const url = new URL(endpoint);
    const data = Buffer.from(JSON.stringify(payload), 'utf8');
    const transport = url.protocol === 'https:' ? https : http;
    return new Promise((resolve, reject) => {
        const req = transport.request({
            method: 'POST',
            hostname: url.hostname,
            port: url.port,
            path: `${url.pathname}${url.search}`,
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': data.length
            },
            timeout: timeoutMs
        }, (res) => {
            const chunks = [];
            res.on('data', (chunk) => chunks.push(chunk));
            res.on('end', () => {
                const body = Buffer.concat(chunks).toString('utf8');
                if (!res.statusCode || res.statusCode < 200 || res.statusCode >= 300) {
                    reject(new Error(`Backend returned HTTP ${res.statusCode}: ${body}`));
                    return;
                }
                try {
                    resolve(JSON.parse(body));
                }
                catch (error) {
                    reject(new Error(`Invalid JSON response from backend: ${error.message}`));
                }
            });
        });
        req.on('timeout', () => {
            req.destroy(new Error(`Backend request timed out after ${timeoutMs} ms.`));
        });
        req.on('error', (error) => {
            if (error.code === 'ECONNREFUSED') {
                reject(new Error('Backend not running. Start python scripts/start_backend.py'));
                return;
            }
            reject(error);
        });
        req.write(data);
        req.end();
    });
}
function getJson(endpoint, timeoutMs) {
    const url = new URL(endpoint);
    const transport = url.protocol === 'https:' ? https : http;
    return new Promise((resolve, reject) => {
        const req = transport.request({
            method: 'GET',
            hostname: url.hostname,
            port: url.port,
            path: `${url.pathname}${url.search}`,
            timeout: timeoutMs
        }, (res) => {
            const chunks = [];
            res.on('data', (chunk) => chunks.push(chunk));
            res.on('end', () => {
                const body = Buffer.concat(chunks).toString('utf8');
                if (!res.statusCode || res.statusCode < 200 || res.statusCode >= 300) {
                    reject(new Error(`Backend returned HTTP ${res.statusCode}: ${body}`));
                    return;
                }
                try {
                    resolve(JSON.parse(body));
                }
                catch (error) {
                    reject(new Error(`Invalid JSON response from backend: ${error.message}`));
                }
            });
        });
        req.on('timeout', () => {
            req.destroy(new Error(`Backend request timed out after ${timeoutMs} ms.`));
        });
        req.on('error', (error) => {
            if (error.code === 'ECONNREFUSED') {
                reject(new Error('Backend not running. Start python scripts/start_backend.py'));
                return;
            }
            reject(error);
        });
        req.end();
    });
}
function validateGenerateResponse(value) {
    if (!value || typeof value !== 'object') {
        throw new Error('Invalid backend response: expected an object.');
    }
    const response = value;
    if (typeof response.generated_code !== 'string') {
        throw new Error('Invalid backend response: generated_code is missing or not a string.');
    }
    if (typeof response.explanation !== 'string') {
        throw new Error('Invalid backend response: explanation is missing or not a string.');
    }
    if (!response.validation || typeof response.validation !== 'object') {
        throw new Error('Invalid backend response: validation is missing.');
    }
    if (!Array.isArray(response.rag_sources)) {
        throw new Error('Invalid backend response: rag_sources is missing or not an array.');
    }
    return response;
}
function validateRagStatusResponse(value) {
    if (!value || typeof value !== 'object') {
        throw new Error('Invalid RAG status response: expected an object.');
    }
    const response = value;
    if (typeof response.project_id !== 'string' || typeof response.indexed !== 'boolean') {
        throw new Error('Invalid RAG status response.');
    }
    return response;
}
function validateRagIndexResponse(value) {
    if (!value || typeof value !== 'object') {
        throw new Error('Invalid RAG index response: expected an object.');
    }
    const response = value;
    if (response.status !== 'success' || typeof response.project_id !== 'string') {
        throw new Error('Invalid RAG index response.');
    }
    return response;
}
function validateRagResetResponse(value) {
    if (!value || typeof value !== 'object') {
        throw new Error('Invalid RAG reset response: expected an object.');
    }
    const response = value;
    if (response.status !== 'success' || typeof response.project_id !== 'string') {
        throw new Error('Invalid RAG reset response.');
    }
    return response;
}
function apiBaseFromGenerateEndpoint(endpoint) {
    const url = new URL(endpoint);
    const pathname = url.pathname.replace(/\/$/, '');
    if (pathname.endsWith('/generate')) {
        url.pathname = pathname.slice(0, -'/generate'.length);
    }
    return url.toString().replace(/\/$/, '');
}
//# sourceMappingURL=apiClient.js.map