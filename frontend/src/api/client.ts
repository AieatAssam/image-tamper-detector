import axios, { AxiosError } from 'axios';
import type { AnalysisResponse, DetectorListResponse } from '../types/api';

export const MAX_UPLOAD_BYTES = 12_000_000;
export const MAX_IMAGE_PIXELS = 50_000_000;
export const ACCEPTED_FORMATS = ['JPEG', 'PNG', 'WEBP', 'TIFF'] as const;

export type AnalysisErrorKind =
  | 'file-too-large'
  | 'unsupported-format'
  | 'rate-limited'
  | 'server-error'
  | 'offline'
  | 'aborted';

export class AnalysisError extends Error {
  constructor(
    message: string,
    public readonly kind: AnalysisErrorKind,
    public readonly retryable = false,
  ) {
    super(message);
    this.name = 'AnalysisError';
  }
}

export class FileTooLarge extends AnalysisError {
  constructor(limit = MAX_UPLOAD_BYTES) {
    super(
      `That file is too large. Choose an image under ${Math.round(limit / 1_000_000)} MB.`,
      'file-too-large',
    );
    this.name = 'FileTooLarge';
  }
}

export class UnsupportedFormat extends AnalysisError {
  constructor() {
    super(
      `That file format is not supported. Use ${ACCEPTED_FORMATS.join(', ')}.`,
      'unsupported-format',
    );
    this.name = 'UnsupportedFormat';
  }
}

export class RateLimited extends AnalysisError {
  constructor(retryAfter: string | undefined) {
    super(
      retryAfter
        ? `Too many analyses. Try again in ${retryAfter}.`
        : 'Too many analyses. Try again shortly.',
      'rate-limited',
      true,
    );
    this.name = 'RateLimited';
  }
}

export class ServerError extends AnalysisError {
  constructor() {
    super(
      'The analysis service is having trouble. You can retry in a moment.',
      'server-error',
      true,
    );
    this.name = 'ServerError';
  }
}

export class Offline extends AnalysisError {
  constructor() {
    super(
      'The analysis service cannot be reached. Check your connection and try again.',
      'offline',
      true,
    );
    this.name = 'Offline';
  }
}

export class RequestAborted extends AnalysisError {
  constructor() {
    super('The analysis was cancelled.', 'aborted');
    this.name = 'RequestAborted';
  }
}

function mapAxiosError(error: unknown): AnalysisError {
  if (axios.isCancel(error)) return new RequestAborted();
  if (!axios.isAxiosError(error)) return new Offline();

  const response = error.response;
  if (!response) return new Offline();
  if (response.status === 413) return new FileTooLarge();
  if (response.status === 415) return new UnsupportedFormat();
  if (response.status === 429) return new RateLimited(response.headers['retry-after']);
  if (response.status >= 500) return new ServerError();
  return new AnalysisError(
    'The request could not be completed. Check the image and try again.',
    'server-error',
  );
}

export interface AnalyzeOptions {
  detectors?: string[];
  includeMaps?: boolean;
  signal?: AbortSignal;
  onUploadProgress?: (percent: number) => void;
}

export async function analyze(file: File, options: AnalyzeOptions = {}): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);
  if (options.detectors?.length) formData.append('detectors', options.detectors.join(','));
  formData.append('include_maps', String(options.includeMaps ?? true));

  try {
    const response = await axios.post<AnalysisResponse>('/api/v1/analyze', formData, {
      signal: options.signal,
      timeout: 120_000,
      onUploadProgress: (event) => {
        if (event.total) options.onUploadProgress?.(Math.round((event.loaded / event.total) * 100));
      },
    });
    return response.data;
  } catch (error) {
    throw mapAxiosError(error);
  }
}

export async function getDetectors(signal?: AbortSignal): Promise<DetectorListResponse> {
  try {
    const response = await axios.get<DetectorListResponse>('/api/v1/detectors', {
      signal,
      timeout: 15_000,
    });
    return response.data;
  } catch (error) {
    throw mapAxiosError(error);
  }
}

export function validateFileSize(file: File): void {
  if (file.size > MAX_UPLOAD_BYTES) throw new FileTooLarge();
}

export async function validateImageDimensions(file: File): Promise<void> {
  validateFileSize(file);
  if (!file.type.startsWith('image/')) throw new UnsupportedFormat();
  if (typeof createImageBitmap !== 'function') return;
  const bitmap = await createImageBitmap(file);
  try {
    if (bitmap.width * bitmap.height > MAX_IMAGE_PIXELS) {
      throw new AnalysisError(
        'That image has too many pixels to inspect safely.',
        'file-too-large',
      );
    }
  } finally {
    bitmap.close();
  }
}

export function errorMessage(error: unknown): string {
  if (error instanceof AnalysisError) return error.message;
  if (error instanceof AxiosError) return mapAxiosError(error).message;
  return 'The analysis could not be completed. Try another image.';
}
