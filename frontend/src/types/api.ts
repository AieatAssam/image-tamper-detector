export type Verdict =
  | 'manipulated'
  | 'likely_manipulated'
  | 'inconclusive'
  | 'likely_authentic'
  | 'authentic';

export type DetectorState = 'applicable' | 'not_applicable' | 'error';

export interface DetectorInfo {
  id: string;
  name: string;
  family: string;
  applicable_formats: string[];
  produces_map: boolean;
  description: string;
  limitations: string[];
  enabled: boolean;
}

export interface DetectorListResponse {
  detectors: DetectorInfo[];
}

export interface DetectorResult {
  id: string;
  state: DetectorState;
  flagged: boolean | null;
  score: number | null;
  threshold: number;
  reason: string;
  metrics: Record<string, number | null>;
  visualization_png_base64: string | null;
  duration_ms: number;
  error: string | null;
}

export interface FusionContribution {
  id: string;
  weight: number;
  signed_contribution: number;
}

export interface FusionResult {
  method: string;
  contributions: FusionContribution[];
  calibration_version: string;
}

export interface AnalysisResponse {
  verdict: Verdict;
  score: number;
  summary: string;
  image: {
    width: number;
    height: number;
    format: string;
    bytes: number;
    sha256: string;
  };
  detectors: DetectorResult[];
  fusion: FusionResult;
  warnings: string[];
}
