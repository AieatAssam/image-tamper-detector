export interface AnalysisDetails {
  method: string;
  description: string;
  edge_discontinuity?: number;
  texture_variance?: number;
  noise_consistency?: number;
  compression_artifacts?: number;
  matching_proportion?: number;
}

export interface AnalysisResponse {
  is_tampered: boolean;
  confidence_score: number;
  analysis_type: 'ELA' | 'PRNU' | 'Entropy';
  visualization_base64: string;
  details: AnalysisDetails;
}

export interface CombinedAnalysisResponse {
  is_tampered: boolean;
  confidence_score: number;
  ela_result?: AnalysisResponse;
  prnu_result?: AnalysisResponse;
  entropy_result?: AnalysisResponse;
  ela_visualization_base64?: string;
  prnu_visualization_base64?: string;
  entropy_visualization_base64?: string;
} 