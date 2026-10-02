/** percent-nlu — tiny on-device NLU for percentage questions (English, Portuguese, Spanish). */

export type Lang = "en" | "pt" | "es";
export type Tool =
  | "percent_of" | "what_percent" | "percent_increase" | "percent_decrease" | "apply_increase" | "apply_decrease";
export type Status = "call" | "low_confidence" | "incomplete" | "out_of_scope";
export type Role = "rate" | "base" | "part" | "whole" | "initial" | "final" | "none";

export interface NumberSlot {
  slot: string;          // "N0".."N3"
  surface: string;       // text as written ("1.200", "twenty-five", "15%")
  value: number;
  has_percent: boolean;
}

export interface Prediction {
  lang: Lang;
  status: Status;        // execute only "call"; confirm "low_confidence"; ask for the missing value on "incomplete"
  tool: Tool | null;
  intent: Tool | "out_of_scope";
  args: Partial<Record<Exclude<Role, "none">, number>>;
  roles: Record<string, Role>;   // slot → role
  missing: string[];             // roles not found in the text (status "incomplete")
  result?: number | null;        // computed by the tool when status is "call" or "low_confidence"
  formula?: string | null;
  warnings?: string[];           // e.g. "direction_mismatch", "rate_over_100"
  confidence: number;            // calibrated, 0..1
  tau: number;                   // confidence threshold used for "call"
  top_intents: [string, number][];
  masked: string;                // normalized text with numbers masked as [N0]..[N3]
  numbers: NumberSlot[];
  timing: { normalize_ms: number; featurize_ms: number; forward_ms: number; decode_ms: number; total_ms: number };
}

export declare class NLU {
  /** Loads model/manifest.json + model/weights.bin (browser: fetch; Node: file path or file:// URL). */
  static load(base?: string | URL, fetchFn?: typeof fetch): Promise<NLU>;
  constructor(manifest: unknown, weights: ArrayBuffer);
  readonly name: string;
  readonly tau: number;
  loadTiming?: { fetch_ms: number; init_ms: number; bytes: number };
  /** lang: pass the page/user language when known; omitted → detected from function words. */
  predict(text: string, lang?: Lang): Prediction;
}

export declare const TOOLS: Record<Tool, (args: Record<string, number>) => { result: number; formula: string; warnings: string[] }>;
export declare function detectLang(text: string, lexicons: Record<string, unknown>, langs: Lang[]): Lang;
