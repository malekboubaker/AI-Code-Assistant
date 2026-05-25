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
  | 'project_explain';

export interface GenerateRequest {
  instruction: string;
  code: string;
  language: string;
  task: BackendTask | null;
  file_path?: string;
  project_path?: string;
  use_rag: boolean;
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

export interface GenerateResponse {
  task: BackendTask;
  language: string;
  generated_code: string;
  explanation: string;
  diff?: string | null;
  used_rag: boolean;
  rag_sources: RagSource[];
  validation: ValidationResult;
  metadata: Record<string, unknown>;
}

export class ApiClient {
  constructor(
    private readonly endpoint: string,
    private readonly timeoutMs: number
  ) {
    assertLocalhostEndpoint(endpoint);
  }

  async generate(request: GenerateRequest): Promise<GenerateResponse> {
    const response = await postJson(this.endpoint, request, this.timeoutMs);
    return validateGenerateResponse(response);
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
