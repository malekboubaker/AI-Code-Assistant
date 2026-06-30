import * as http from 'node:http';
import * as https from 'node:https';

export type BackendTask =
  | 'auto_complete'
  | 'code_gen'
  | 'bug_detection'
  | 'bug_fix'
  | 'refactoring'
  | 'perf_opt'
  | 'test_gen'
  | 'explain'
  | 'project_explain'
  | 'file_explain'
  | 'compare';

export interface GenerateRequest {
  instruction: string;
  code: string;
  language: string;
  task: BackendTask | null;
  file_path?: string;
  project_path?: string;
  has_selection?: boolean;
  surrounding_context?: string;
  chat_history?: ChatHistoryMessage[];
  use_rag: boolean;
  response_id?: string;
}

export interface ChatHistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ValidationResult {
  valid: boolean;
  syntax_valid?: boolean | null;
  tests_passed?: boolean | null;
  warnings: string[];
}

export interface RagSource {
  content: string;
  score: number;
  language?: string;
  file_path?: string;
  start_line?: number;
  end_line?: number;
  chunk_type?: string;
  symbol_name?: string;
  metadata?: Record<string, unknown>;
}

export interface WorkspaceEdit {
  file_path: string;
  reason?: string;
  original_content?: string;
  new_content: string;
}

export interface GenerateResponse {
  task: BackendTask;
  language: string;
  generated_code: string;
  explanation: string;
  diff?: string | null;
  used_rag: boolean;
  rag_sources: RagSource[];
  validation: ValidationResult;
  edits?: WorkspaceEdit[];
  metadata: Record<string, unknown>;
}

export interface RagStatusResponse {
  project_id: string;
  project_path: string;
  indexed: boolean;
  project_map_exists: boolean;
  point_count: number;
  last_indexed?: string | null;
  detected_languages: Record<string, number>;
  frameworks: string[];
  entry_points: string[];
  qdrant_collection: string;
  qdrant_ready?: boolean;
}

export type RagIndexMode = 'incremental' | 'full';

export interface RagIndexResponse {
  status: string;
  project_id: string;
  files_scanned: number;
  files_indexed: number;
  files_skipped: number;
  chunks_created: number;
  chunks_stored: number;
  project_map_exists: boolean;
  duration_ms: number;
}

export interface RagResetResponse {
  status: string;
  project_id: string;
  deleted_points: number;
}

export class ApiClient {
  private readonly apiBaseUrl: string;

  constructor(
    private readonly endpoint: string,
    private readonly timeoutMs: number
  ) {
    assertLocalhostEndpoint(endpoint);
    this.apiBaseUrl = apiBaseFromGenerateEndpoint(endpoint);
  }

  async generate(request: GenerateRequest): Promise<GenerateResponse> {
    const response = await postJson(this.endpoint, request, this.timeoutMs);
    return validateGenerateResponse(response);
  }

  streamGenerate(request: GenerateRequest, onEvent: (event: any) => void): Promise<GenerateResponse> {
    const url = new URL(`${this.apiBaseUrl}/generate/stream`);
    const data = Buffer.from(JSON.stringify(request), 'utf8');
    const transport = url.protocol === 'https:' ? https : http;

    return new Promise((resolve, reject) => {
      const req = transport.request(
        {
          method: 'POST',
          hostname: url.hostname,
          port: url.port,
          path: `${url.pathname}${url.search}`,
          headers: {
            'Content-Type': 'application/json',
            'Content-Length': data.length,
          },
          timeout: this.timeoutMs,
        },
        (res) => {
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
              if (!line.trim()) continue;
              try {
                const event = JSON.parse(line);
                if (event.type === 'result') {
                  try {
                    resolve(validateGenerateResponse(event.payload));
                  } catch (valErr) {
                    reject(valErr);
                  }
                } else {
                  onEvent(event);
                }
              } catch (e) {
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
                  } catch (valErr) {
                    reject(valErr);
                  }
                } else onEvent(event);
              } catch (e) {
                reject(new Error("Stream ended with incomplete or invalid JSON: " + buffer));
              }
            } else {
              reject(new Error("Stream ended without a final result payload."));
            }
          });
        }
      );

      req.on('error', reject);
      req.on('timeout', () => {
        req.destroy();
        reject(new Error(`Backend request timed out after ${this.timeoutMs} ms.`));
      });
      
      req.write(data);
      req.end();
    });
  }

  async cancelRequest(responseId: string): Promise<void> {
    await postJson(`${this.apiBaseUrl}/cancel/${encodeURIComponent(responseId)}`, {}, this.timeoutMs);
  }

  async getRagStatus(projectPath: string): Promise<RagStatusResponse> {
    const url = `${this.apiBaseUrl}/rag/status?project_path=${encodeURIComponent(projectPath)}`;
    const response = await getJson(url, this.timeoutMs);
    return validateRagStatusResponse(response);
  }

  async indexProject(projectPath: string, mode: RagIndexMode): Promise<RagIndexResponse> {
    const response = await postJson(`${this.apiBaseUrl}/rag/index`, { project_path: projectPath, mode }, this.timeoutMs);
    return validateRagIndexResponse(response);
  }

  async resetProjectIndex(projectPath: string): Promise<RagResetResponse> {
    const response = await postJson(`${this.apiBaseUrl}/rag/reset`, { project_path: projectPath }, this.timeoutMs);
    return validateRagResetResponse(response);
  }
}

function assertLocalhostEndpoint(endpoint: string): void {
  const url = new URL(endpoint);
  const allowedHosts = new Set(['localhost', '127.0.0.1', '::1']);
  if (!allowedHosts.has(url.hostname)) {
    throw new Error(`Refusing non-local backend URL: ${endpoint}`);
  }
  if (url.protocol !== 'http:') {
    throw new Error(`Only local http:// backend URLs are allowed: ${endpoint}`);
  }
}

function postJson(endpoint: string, payload: unknown, timeoutMs: number): Promise<unknown> {
  const url = new URL(endpoint);
  const data = Buffer.from(JSON.stringify(payload), 'utf8');
  const transport = url.protocol === 'https:' ? https : http;

  return new Promise((resolve, reject) => {
    const req = transport.request(
      {
        method: 'POST',
        hostname: url.hostname,
        port: url.port,
        path: `${url.pathname}${url.search}`,
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': data.length
        },
        timeout: timeoutMs
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (chunk: Buffer) => chunks.push(chunk));
        res.on('end', () => {
          const body = Buffer.concat(chunks).toString('utf8');
          if (!res.statusCode || res.statusCode < 200 || res.statusCode >= 300) {
            reject(new Error(`Backend returned HTTP ${res.statusCode}: ${body}`));
            return;
          }
          try {
            resolve(JSON.parse(body));
          } catch (error) {
            reject(new Error(`Invalid JSON response from backend: ${(error as Error).message}`));
          }
        });
      }
    );

    req.on('timeout', () => {
      req.destroy(new Error(`Backend request timed out after ${timeoutMs} ms.`));
    });
    req.on('error', (error: NodeJS.ErrnoException) => {
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

function getJson(endpoint: string, timeoutMs: number): Promise<unknown> {
  const url = new URL(endpoint);
  const transport = url.protocol === 'https:' ? https : http;

  return new Promise((resolve, reject) => {
    const req = transport.request(
      {
        method: 'GET',
        hostname: url.hostname,
        port: url.port,
        path: `${url.pathname}${url.search}`,
        timeout: timeoutMs
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (chunk: Buffer) => chunks.push(chunk));
        res.on('end', () => {
          const body = Buffer.concat(chunks).toString('utf8');
          if (!res.statusCode || res.statusCode < 200 || res.statusCode >= 300) {
            reject(new Error(`Backend returned HTTP ${res.statusCode}: ${body}`));
            return;
          }
          try {
            resolve(JSON.parse(body));
          } catch (error) {
            reject(new Error(`Invalid JSON response from backend: ${(error as Error).message}`));
          }
        });
      }
    );

    req.on('timeout', () => {
      req.destroy(new Error(`Backend request timed out after ${timeoutMs} ms.`));
    });
    req.on('error', (error: NodeJS.ErrnoException) => {
      if (error.code === 'ECONNREFUSED') {
        reject(new Error('Backend not running. Start python scripts/start_backend.py'));
        return;
      }
      reject(error);
    });
    req.end();
  });
}

function validateGenerateResponse(value: unknown): GenerateResponse {
  if (!value || typeof value !== 'object') {
    throw new Error('Invalid backend response: expected an object.');
  }
  const response = value as Partial<GenerateResponse>;
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
  return response as GenerateResponse;
}

function validateRagStatusResponse(value: unknown): RagStatusResponse {
  if (!value || typeof value !== 'object') {
    throw new Error('Invalid RAG status response: expected an object.');
  }
  const response = value as Partial<RagStatusResponse>;
  if (typeof response.project_id !== 'string' || typeof response.indexed !== 'boolean') {
    throw new Error('Invalid RAG status response.');
  }
  return response as RagStatusResponse;
}

function validateRagIndexResponse(value: unknown): RagIndexResponse {
  if (!value || typeof value !== 'object') {
    throw new Error('Invalid RAG index response: expected an object.');
  }
  const response = value as Partial<RagIndexResponse>;
  if (response.status !== 'success' || typeof response.project_id !== 'string') {
    throw new Error('Invalid RAG index response.');
  }
  return response as RagIndexResponse;
}

function validateRagResetResponse(value: unknown): RagResetResponse {
  if (!value || typeof value !== 'object') {
    throw new Error('Invalid RAG reset response: expected an object.');
  }
  const response = value as Partial<RagResetResponse>;
  if (response.status !== 'success' || typeof response.project_id !== 'string') {
    throw new Error('Invalid RAG reset response.');
  }
  return response as RagResetResponse;
}

function apiBaseFromGenerateEndpoint(endpoint: string): string {
  const url = new URL(endpoint);
  const pathname = url.pathname.replace(/\/$/, '');
  if (pathname.endsWith('/generate')) {
    url.pathname = pathname.slice(0, -'/generate'.length);
  }
  return url.toString().replace(/\/$/, '');
}
