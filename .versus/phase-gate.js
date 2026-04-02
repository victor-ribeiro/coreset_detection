#!/usr/bin/env node
"use strict";
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));

// src/state/engine.ts
var fs = __toESM(require("fs"));
var path = __toESM(require("path"));
var import_crypto = require("crypto");

// src/state/types.ts
function createDefaultState(name, description) {
  return {
    agent: "claude",
    projectName: name,
    projectDescription: description,
    projectSpec: { stack: [], outOfScope: [], patterns: [], constraints: [] },
    currentPhase: 0,
    currentIteration: 1,
    phase0Score: null,
    phase0Breakdown: null,
    decisions: [],
    exitCriteria: [],
    safeguards: [
      { id: "S0", status: "ok" },
      { id: "S1", status: "ok" },
      { id: "S2", status: "ok" },
      { id: "S3", status: "ok" },
      { id: "S4", status: "ok" },
      { id: "S5", status: "ok" },
      { id: "S6", status: "ok" },
      { id: "S7", status: "ok" }
    ],
    loopCounter: { pattern: "", count: 0, firstSeen: "", lastSeen: "" },
    createdAt: (/* @__PURE__ */ new Date()).toISOString(),
    updatedAt: (/* @__PURE__ */ new Date()).toISOString(),
    history: []
  };
}

// src/rules/phases.ts
var EXIT_CRITERIA = [
  // Phase 0
  { phase: 0, criterion: "score_90", description: "Score >= 90/100", required: true },
  { phase: 0, criterion: "user_confirmed", description: "User confirmed synthesis", required: true },
  { phase: 0, criterion: "ambiguities_zero", description: "Ambiguities = 0 (or accepted)", required: true },
  { phase: 0, criterion: "use_cases_complete", description: "Use cases complete", required: true },
  { phase: 0, criterion: "vocabulary_agreed", description: "Vocabulary agreed upon", required: true },
  { phase: 0, criterion: "out_of_scope_clear", description: "Out of scope clear", required: true },
  { phase: 0, criterion: "tech_feasibility", description: "Target platform tech feasibility VERIFIED: fundamental capabilities confirmed (not assumed). In porting projects: does the destination platform support essential mechanisms?", required: true },
  { phase: 0, criterion: "implementation_feasibility", description: "Components evaluated Tier 1/2/3. If complex Tier 3 \u2192 PoC before Phase 1", required: true },
  { phase: 0, criterion: "technical_scientific_research", description: "If specialized domain: complete technical AND scientific research (F.2) \u2014 papers, algorithms, parameters with source", required: false },
  { phase: 0, criterion: "specs_populated", description: "specs/ populated with technical and scientific references", required: true },
  // Phase 1
  { phase: 1, criterion: "patterns_defined", description: "Patterns/principles defined and confirmed", required: true },
  { phase: 1, criterion: "modules_responsibility", description: "Each module with clear responsibility", required: true },
  { phase: 1, criterion: "interfaces_defined", description: "Interfaces defined (signatures, I/O types)", required: true },
  { phase: 1, criterion: "dependencies_explicit", description: "Dependencies between modules explicit", required: true },
  { phase: 1, criterion: "assumptions_listed", description: "Assumptions listed", required: true },
  { phase: 1, criterion: "architecture_doc", description: "Architecture doc fits in ~2k tokens", required: true },
  { phase: 1, criterion: "tech_100_scope", description: "Tech supports 100% of scope, tech assumptions verified", required: true },
  { phase: 1, criterion: "modules_adherent_patterns", description: "Modules adherent to chosen patterns", required: true },
  // Phase 2
  { phase: 2, criterion: "activated_lenses_recorded", description: "Activated conditional lenses recorded via record_decision(category='architecture') with justification for each non-activated lens \u2014 includes 8 situational lenses and 4 Domain Transfer Lenses (Control Engineering, Game Theory, Linguistics/Grammar, Mechanical Engineering)", required: true },
  { phase: 2, criterion: "lenses_applied", description: "All 7 universal lenses + all activated conditional lenses applied to each module (no lens skipped \u2014 absence of findings is a valid result)", required: true },
  { phase: 2, criterion: "coverage_matrix", description: "Coverage matrix modules x lenses (all applied lenses as columns)", required: true },
  { phase: 2, criterion: "criticals_identified", description: "Critical findings identified and classified", required: true },
  { phase: 2, criterion: "concentration_analyzed", description: "Concentration analysis performed (by module and by lens)", required: true },
  // Phase 3
  { phase: 3, criterion: "criticals_addressed", description: "All critical findings addressed", required: true },
  { phase: 3, criterion: "important_decided", description: "All important findings with decision", required: true },
  { phase: 3, criterion: "scope_preserved", description: "Phase 0 scope preserved (anti-scope-creep)", required: true },
  { phase: 3, criterion: "architecture_simplified", description: "Architecture simpler than previous", required: true },
  // Phase 4
  { phase: 4, criterion: "exit_criteria_p2p3", description: "Exit criteria Phases 2-3 verified", required: true },
  { phase: 4, criterion: "safeguards_s1_s5", description: "Safeguards S1-S5 met", required: true },
  // Phase 5
  { phase: 5, criterion: "all_modules", description: "All modules implemented", required: true },
  { phase: 5, criterion: "specs_consulted", description: "specs/ consulted before each module", required: true },
  { phase: 5, criterion: "s6_applied", description: "S6 applied (Tier 1/2/3 per module)", required: true },
  { phase: 5, criterion: "ui_runnable", description: "UI implemented and accessible for manual testing, or N/A documented for backend-only projects", required: true },
  // Phase 6
  { phase: 6, criterion: "tests_passing", description: "100% tests passing", required: true },
  { phase: 6, criterion: "manual_testing", description: "Manual exploratory testing performed", required: true },
  { phase: 6, criterion: "edge_cases", description: "Edge cases tested", required: true },
  // Phase 7
  { phase: 7, criterion: "specs_updated", description: "specs/ updated with results", required: true },
  { phase: 7, criterion: "lessons_documented", description: "Lessons learned documented", required: true },
  { phase: 7, criterion: "human_feedback", description: "Human feedback collected", required: true }
];
var CRITERION_ID_MIGRATION = {
  "usuario_confirmou": "user_confirmed",
  "ambiguidades_zero": "ambiguities_zero",
  "casos_uso_completos": "use_cases_complete",
  "vocabulario_acordado": "vocabulary_agreed",
  "fora_escopo_claro": "out_of_scope_clear",
  "viabilidade_tech": "tech_feasibility",
  "viabilidade_implementacao": "implementation_feasibility",
  "pesquisa_tecnica_cientifica": "technical_scientific_research",
  "specs_populado": "specs_populated",
  "padroes_definidos": "patterns_defined",
  "modulos_responsabilidade": "modules_responsibility",
  "interfaces_definidas": "interfaces_defined",
  "dependencias_explicitas": "dependencies_explicit",
  "premissas_listadas": "assumptions_listed",
  "doc_arquitetura": "architecture_doc",
  "tech_100_escopo": "tech_100_scope",
  "modulos_aderentes_padroes": "modules_adherent_patterns",
  "lentes_aplicadas": "lenses_applied",
  "matriz_cobertura": "coverage_matrix",
  "criticos_identificados": "criticals_identified",
  "concentracao_analisada": "concentration_analyzed",
  "criticos_endere\xE7ados": "criticals_addressed",
  "criticos_enderecados": "criticals_addressed",
  "importantes_decididos": "important_decided",
  "escopo_preservado": "scope_preserved",
  "arquitetura_simplificada": "architecture_simplified",
  "exit_criteria_23": "exit_criteria_p2p3",
  "salvaguardas_s1_s5": "safeguards_s1_s5",
  "todos_modulos": "all_modules",
  "specs_consultado": "specs_consulted",
  "s6_aplicado": "s6_applied",
  "testes_passando": "tests_passing",
  "teste_manual": "manual_testing",
  "specs_atualizado": "specs_updated",
  "licoes_documentadas": "lessons_documented",
  "feedback_humano": "human_feedback"
};
function getExitCriteriaForPhase(phase) {
  return EXIT_CRITERIA.filter((c) => c.phase === phase);
}
function isValidTransition(from, to) {
  if (to === from + 1) return true;
  if (from === 3 && to === 2) return true;
  if (from === 2 && to === 3) return true;
  return false;
}

// src/rules/safeguards.ts
var SAFEGUARD_DEFINITIONS = [
  {
    id: "S0",
    name: "Problem Convergence",
    description: "Never advance to Phase 1 without score >= 90/100. Wrong problem costs 100x.",
    applicablePhases: [0]
  },
  {
    id: "S1",
    name: "Anti-Bug",
    description: "Simplification never introduces bugs. Features maintained after each Phase 3.",
    applicablePhases: [3]
  },
  {
    id: "S2",
    name: "Stopping Criterion",
    description: "Stopping criterion belongs to the USER, not the AI.",
    applicablePhases: [0, 1, 2, 3, 4, 5, 6, 7]
  },
  {
    id: "S3",
    name: "Premature Convergence Cost",
    description: "Stopping early = -400% to -600% ROI. Prefer iterating when in doubt.",
    applicablePhases: [0, 2, 3]
  },
  {
    id: "S4",
    name: "Explicit Verification (mandatory human-AV)",
    description: "Human-AV is irreplaceable at each gate. Automated tests verify formalizable properties. Semantic adequacy, usability, and domain correctness REQUIRE human judgment. P0: human validates synthesis. P2: human arbitrated trade-offs. P4: human confirms convergence. P6: NEVER assume tests passed \u2014 execute and verify + mandatory manual exploratory testing.",
    applicablePhases: [0, 2, 4, 6]
  },
  {
    id: "S5",
    name: "Scope Preservation",
    description: "Phase 2-3 operates WITHIN Phase 0 scope. Sub-rules: 5.1 Scope belongs to the user \u2014 Phase 2-3 suggests, never decides changes. 5.2 If not requested, don't add. If useful, document as v2.0 suggestion. 5.3 Detector: 'If the user compared V(N) with Phase 0, would they say this isn't what I asked for?' If yes \u2192 scope violated.",
    applicablePhases: [2, 3]
  },
  {
    id: "S6",
    name: "Don't Reimplement What Already Exists",
    description: "Tier 1: mature lib \u2192 USE IT. Tier 2: algorithm with ref \u2192 PORT literally (same structure, same names, test against same inputs). Tier 3: neither of the above. If complex domain \u2192 PoC (~2h max). Immediate STOP if: creating heuristics for problem with known solution, debugging complex logic from scratch, trial-and-error on something deterministic, or >2 iterations on the same module. Checklist per module: mature lib? \u2192 if not, why? \u2192 documented algorithm? \u2192 portable ref? \u2192 decision.",
    applicablePhases: [5, 6]
  },
  {
    id: "S7",
    name: "Sequence Discipline",
    description: "After each file: mark completed, identify next, announce progress, start immediately.",
    applicablePhases: [5]
  }
];
function getSafeguardDefinition(id) {
  return SAFEGUARD_DEFINITIONS.find((s) => s.id === id);
}
function getSafeguardsForPhase(phase) {
  return SAFEGUARD_DEFINITIONS.filter((s) => s.applicablePhases.includes(phase));
}
function validateSafeguard(id, state) {
  const def = getSafeguardDefinition(id);
  if (!def) {
    return { id, status: "ok", details: `Safeguard ${id} not found.` };
  }
  switch (id) {
    case "S0":
      return validateS0(state);
    case "S1":
      return validateS1(state);
    case "S5":
      return validateS5(state);
    case "S6":
      return validateS6(state);
    case "S7":
      return validateS7(state);
    default:
      return {
        id,
        status: "ok",
        details: `${def.name}: behavioral check \u2014 requires agent attention. ${def.description}`
      };
  }
}
function validateS0(state) {
  if (state.currentPhase === 0) {
    return { id: "S0", status: "ok", details: "Still in Phase 0. Current score: " + (state.phase0Score ?? "not evaluated") };
  }
  if (state.phase0Score === null || state.phase0Score < 90) {
    return {
      id: "S0",
      status: "violated",
      details: `Phase 0 score ${state.phase0Score ?? "null"} < 90. Should not have advanced.`
    };
  }
  return { id: "S0", status: "ok", details: `Phase 0 score ${state.phase0Score}/100. OK.` };
}
function validateS1(state) {
  if (state.currentPhase !== 3) {
    return { id: "S1", status: "ok", details: "Not in Phase 3." };
  }
  return {
    id: "S1",
    status: "warning",
    details: "Phase 3 active: verify that simplification did not introduce bugs. Features must be maintained."
  };
}
function validateS5(state) {
  if (state.currentPhase !== 2 && state.currentPhase !== 3) {
    return { id: "S5", status: "ok", details: "Not in Phase 2-3." };
  }
  return {
    id: "S5",
    status: "warning",
    details: "Phase 2-3 active: Phase 0 scope must be preserved. Don't cut requirements, don't add features."
  };
}
function validateS6(state) {
  if (state.currentPhase < 5) {
    return { id: "S6", status: "ok", details: "Not in implementation." };
  }
  return {
    id: "S6",
    status: "warning",
    details: "Implementation active: verify Tier 1/2/3 per module. STOP if creating heuristics for problems with known solutions."
  };
}
function validateS7(state) {
  if (state.currentPhase !== 5) {
    return { id: "S7", status: "ok", details: "Not in Phase 5." };
  }
  return {
    id: "S7",
    status: "warning",
    details: "Phase 5 active: after each file, mark completed and announce progress. Don't start tangential discussions."
  };
}

// src/state/engine.ts
function validatePhaseTransition(from, to, state, exitCriteriaMet) {
  const result = {
    valid: true,
    missingCriteria: [],
    safeguardViolations: []
  };
  if (!isValidTransition(from, to)) {
    result.valid = false;
    result.missingCriteria.push(
      `Transition from Phase ${from} to Phase ${to} is not allowed. Valid transitions: sequential or loop 2\u21943.`
    );
    return result;
  }
  if (to > from) {
    const criteria = getExitCriteriaForPhase(from);
    for (const criterion of criteria) {
      if (criterion.required && !exitCriteriaMet.get(criterion.criterion)) {
        result.valid = false;
        result.missingCriteria.push(
          `[Phase ${from}] ${criterion.criterion}: ${criterion.description}`
        );
      }
    }
  }
  if (from === 0 && to === 1) {
    if (state.phase0Score === null || state.phase0Score < 90) {
      result.valid = false;
      result.missingCriteria.push(
        `Phase 0 score = ${state.phase0Score ?? "null"}. Required >= 90.`
      );
    }
  }
  const safeguards = getSafeguardsForPhase(from);
  for (const safeguard of safeguards) {
    const check = validateSafeguard(safeguard.id, state);
    if (check.status === "violated") {
      result.valid = false;
      result.safeguardViolations.push(`${safeguard.id}: ${check.details}`);
    }
  }
  return result;
}
var StateEngine = class {
  state = null;
  statePath;
  specsPath;
  workspacePath;
  constructor(workspacePath) {
    this.workspacePath = workspacePath;
    this.statePath = path.join(workspacePath, ".versus", "state.json");
    this.specsPath = path.join(workspacePath, "specs");
  }
  getWorkspace() {
    return this.workspacePath;
  }
  // --- Lifecycle ---
  load() {
    try {
      if (fs.existsSync(this.statePath)) {
        const raw = fs.readFileSync(this.statePath, "utf-8");
        this.state = JSON.parse(raw);
        this.migrateCriterionIds();
        return this.state;
      }
    } catch (err) {
    }
    return null;
  }
  /** Migrate Portuguese criterion IDs to English (v0.3.5 → v0.3.6) */
  migrateCriterionIds() {
    if (!this.state) return;
    let changed = false;
    for (const c of this.state.exitCriteria) {
      const newId = CRITERION_ID_MIGRATION[c.criterion];
      if (newId) {
        c.criterion = newId;
        changed = true;
      }
    }
    for (const h of this.state.history) {
      if (h.criteriaMet) {
        h.criteriaMet = h.criteriaMet.map((id) => CRITERION_ID_MIGRATION[id] || id);
      }
    }
    if (changed) {
      this.save();
    }
  }
  save() {
    if (!this.state) return;
    this.state.updatedAt = (/* @__PURE__ */ new Date()).toISOString();
    const dir = path.dirname(this.statePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(this.statePath, JSON.stringify(this.state, null, 2), "utf-8");
  }
  // --- Project ---
  initProject(name, description) {
    this.state = createDefaultState(name, description);
    this.save();
    return this.state;
  }
  // --- Gateway Guard ---
  touchPhaseStateCheck() {
    if (!this.state) return;
    this.state.lastPhaseStateCheck = (/* @__PURE__ */ new Date()).toISOString();
    this.save();
  }
  isContextStale(thresholdMinutes = 30) {
    if (!this.state) return true;
    if (!this.state.lastPhaseStateCheck) return true;
    const last = new Date(this.state.lastPhaseStateCheck).getTime();
    return Date.now() - last > thresholdMinutes * 60 * 1e3;
  }
  // --- Query ---
  getPhaseState() {
    if (!this.state) this.load();
    return this.state;
  }
  getDecisions(phase) {
    if (!this.state) return [];
    if (phase !== void 0) {
      return this.state.decisions.filter((d) => d.phase === phase);
    }
    return this.state.decisions;
  }
  getExitCriteriaState(phase) {
    if (!this.state) return [];
    return this.state.exitCriteria.filter((c) => c.phase === phase);
  }
  getExitCriteriaWithDefs(phase) {
    const defs = getExitCriteriaForPhase(phase);
    const states = this.getExitCriteriaState(phase);
    return defs.map((def) => {
      const state = states.find((s) => s.criterion === def.criterion);
      return {
        criterion: def.criterion,
        description: def.description,
        met: state?.met ?? false,
        details: state?.details
      };
    });
  }
  // --- Transitions ---
  advancePhase(targetPhase) {
    if (!this.state) {
      return { ok: false, error: { message: "Project not initialized.", missingCriteria: [] } };
    }
    const from = this.state.currentPhase;
    const exitCriteriaMet = /* @__PURE__ */ new Map();
    for (const c of this.state.exitCriteria) {
      if (c.phase === from) {
        exitCriteriaMet.set(c.criterion, c.met);
      }
    }
    const validation = validatePhaseTransition(from, targetPhase, this.state, exitCriteriaMet);
    const specsWarnings = [];
    if (targetPhase >= 1) {
      const specsStatus = this.checkSpecsStatus();
      const expectedDirs = this.getExpectedSpecsDirs(from);
      for (const dir of expectedDirs) {
        if (specsStatus[dir] && !specsStatus[dir].populated) {
          specsWarnings.push(`specs/${dir}/ is empty (expected before Phase ${targetPhase})`);
        }
      }
    }
    if (!validation.valid) {
      return {
        ok: false,
        error: {
          message: `Cannot advance from Phase ${from} to Phase ${targetPhase}.`,
          missingCriteria: validation.missingCriteria,
          specsWarnings
        }
      };
    }
    this.state.history.push({
      from,
      to: targetPhase,
      timestamp: (/* @__PURE__ */ new Date()).toISOString(),
      criteriaMet: Array.from(exitCriteriaMet.entries()).filter(([, met]) => met).map(([criterion]) => criterion)
    });
    this.state.currentPhase = targetPhase;
    this.state.currentIteration = 1;
    this.state.loopCounter = { pattern: "", count: 0, firstSeen: "", lastSeen: "" };
    this.save();
    return { ok: true, value: void 0 };
  }
  startIteration(phase) {
    if (!this.state) {
      return { ok: false, error: "Project not initialized." };
    }
    if (this.state.currentPhase !== phase) {
      return { ok: false, error: `Current phase is ${this.state.currentPhase}, not ${phase}.` };
    }
    this.state.currentIteration += 1;
    this.save();
    return {
      ok: true,
      value: {
        phase,
        iterationNumber: this.state.currentIteration
      }
    };
  }
  // --- Recording ---
  recordDecision(phase, category, content) {
    if (!this.state) throw new Error("Project not initialized.");
    const decision = {
      id: (0, import_crypto.randomUUID)(),
      phase,
      category,
      content,
      timestamp: (/* @__PURE__ */ new Date()).toISOString()
    };
    this.state.decisions.push(decision);
    this.save();
    return decision;
  }
  updateProjectSpec(spec) {
    if (!this.state) throw new Error("Project not initialized.");
    if (!this.state.projectSpec) {
      this.state.projectSpec = { stack: [], outOfScope: [], patterns: [], constraints: [] };
    }
    const ps = this.state.projectSpec;
    if (spec.stack) ps.stack = dedupe([...ps.stack, ...spec.stack]);
    if (spec.outOfScope) ps.outOfScope = dedupe([...ps.outOfScope, ...spec.outOfScope]);
    if (spec.patterns) ps.patterns = dedupe([...ps.patterns, ...spec.patterns]);
    if (spec.constraints) ps.constraints = dedupe([...ps.constraints, ...spec.constraints]);
    this.save();
  }
  getProjectSpec() {
    if (!this.state) throw new Error("Project not initialized.");
    return this.state.projectSpec ?? { stack: [], outOfScope: [], patterns: [], constraints: [] };
  }
  updateScore(score, breakdown) {
    if (!this.state) throw new Error("Project not initialized.");
    this.state.phase0Score = score;
    this.state.phase0Breakdown = breakdown;
    this.save();
  }
  markExitCriterion(phase, criterion, met, details) {
    if (!this.state) throw new Error("Project not initialized.");
    const existing = this.state.exitCriteria.find(
      (c) => c.phase === phase && c.criterion === criterion
    );
    if (existing) {
      existing.met = met;
      existing.details = details;
    } else {
      this.state.exitCriteria.push({ phase, criterion, met, details });
    }
    this.save();
  }
  markExitCriteria(entries) {
    if (!this.state) throw new Error("Project not initialized.");
    for (const e of entries) {
      const existing = this.state.exitCriteria.find(
        (c) => c.phase === e.phase && c.criterion === e.criterion
      );
      if (existing) {
        existing.met = e.met;
        existing.details = e.details;
      } else {
        this.state.exitCriteria.push({ phase: e.phase, criterion: e.criterion, met: e.met, details: e.details });
      }
    }
    this.save();
  }
  // --- Safeguards ---
  checkSafeguard(id) {
    if (!this.state) {
      return { id, status: "ok", details: "Project not initialized." };
    }
    const result = validateSafeguard(id, this.state);
    const sg = this.state.safeguards.find((s) => s.id === id);
    if (sg) {
      sg.status = result.status;
      sg.details = result.details;
      sg.lastChecked = (/* @__PURE__ */ new Date()).toISOString();
      this.save();
    }
    return result;
  }
  checkAllSafeguards() {
    if (!this.state) return [];
    const results = getSafeguardsForPhase(this.state.currentPhase).map((s) => validateSafeguard(s.id, this.state));
    for (const result of results) {
      const sg = this.state.safeguards.find((s) => s.id === result.id);
      if (sg) {
        sg.status = result.status;
        sg.details = result.details;
        sg.lastChecked = (/* @__PURE__ */ new Date()).toISOString();
      }
    }
    this.save();
    return results;
  }
  // --- Specs ---
  checkSpecsStatus() {
    const dirs = [
      "references",
      "domain",
      "technical",
      "examples",
      "design",
      "models",
      "datasets",
      "validation",
      "competitors"
    ];
    const status = {};
    for (const dir of dirs) {
      const dirPath = path.join(this.specsPath, dir);
      try {
        if (fs.existsSync(dirPath)) {
          const files = fs.readdirSync(dirPath).filter((f) => f !== "README.md");
          status[dir] = { populated: files.length > 0, fileCount: files.length };
        } else {
          status[dir] = { populated: false, fileCount: 0 };
        }
      } catch {
        status[dir] = { populated: false, fileCount: 0 };
      }
    }
    return status;
  }
  // --- Loop Counter ---
  incrementLoopCounter(pattern) {
    if (!this.state) return { count: 0, blocked: false };
    const normalized = this.normalizeCommandPattern(pattern);
    if (!this.state.loopCounter) {
      this.state.loopCounter = { pattern: "", count: 0, firstSeen: "", lastSeen: "" };
    }
    const counter = this.state.loopCounter;
    if (counter.pattern === normalized && counter.count > 0) {
      counter.count += 1;
      counter.lastSeen = (/* @__PURE__ */ new Date()).toISOString();
    } else {
      this.state.loopCounter = {
        pattern: normalized,
        count: 1,
        firstSeen: (/* @__PURE__ */ new Date()).toISOString(),
        lastSeen: (/* @__PURE__ */ new Date()).toISOString()
      };
    }
    this.save();
    return { count: this.state.loopCounter.count, blocked: this.state.loopCounter.count > 3 };
  }
  resetLoopCounter() {
    if (!this.state) return;
    this.state.loopCounter = { pattern: "", count: 0, firstSeen: "", lastSeen: "" };
    this.save();
  }
  getLoopCounter() {
    if (!this.state || !this.state.loopCounter) return { pattern: "", count: 0 };
    return { pattern: this.state.loopCounter.pattern, count: this.state.loopCounter.count };
  }
  normalizeCommandPattern(cmd) {
    const trimmed = cmd.trim().toLowerCase();
    if (/\b(npm\s+test|npx\s+jest|jest|vitest|mocha)\b/.test(trimmed)) return "test:js";
    if (/\b(pytest|python\s+-m\s+pytest|unittest)\b/.test(trimmed)) return "test:py";
    if (/\b(cargo\s+test)\b/.test(trimmed)) return "test:rust";
    if (/\b(go\s+test)\b/.test(trimmed)) return "test:go";
    if (/\bnpm\s+run\s+build\b/.test(trimmed)) return "build";
    if (/\btsc\b/.test(trimmed)) return "compile";
    return trimmed.substring(0, 50);
  }
  // --- Meta-iteration (v1.0 → v2.0) ---
  startNewCycle() {
    if (!this.state) {
      return { ok: false, error: "Project not initialized." };
    }
    if (this.state.currentPhase !== 7) {
      return { ok: false, error: `Current phase is ${this.state.currentPhase}. New cycle can only start from Phase 7 (Post-Review).` };
    }
    const p7Criteria = this.state.exitCriteria.filter((c) => c.phase === 7);
    const unmet = p7Criteria.filter((c) => !c.met);
    if (unmet.length > 0) {
      return { ok: false, error: `Phase 7 exit criteria not met: ${unmet.map((c) => c.criterion).join(", ")}. Complete Post-Review before starting new cycle.` };
    }
    const cycleCount = this.state.history.filter((h) => h.to === 7).length + 1;
    this.state.history.push({
      from: 7,
      to: 0,
      timestamp: (/* @__PURE__ */ new Date()).toISOString(),
      criteriaMet: p7Criteria.map((c) => c.criterion)
    });
    this.state.currentPhase = 0;
    this.state.currentIteration = 1;
    this.state.phase0Score = null;
    this.state.phase0Breakdown = null;
    this.state.exitCriteria = [];
    this.state.safeguards = [
      { id: "S0", status: "ok" },
      { id: "S1", status: "ok" },
      { id: "S2", status: "ok" },
      { id: "S3", status: "ok" },
      { id: "S4", status: "ok" },
      { id: "S5", status: "ok" },
      { id: "S6", status: "ok" },
      { id: "S7", status: "ok" }
    ];
    this.state.loopCounter = { pattern: "", count: 0, firstSeen: "", lastSeen: "" };
    this.save();
    return { ok: true, value: { cycle: cycleCount + 1 } };
  }
  // --- Private helpers ---
  getExpectedSpecsDirs(phase) {
    switch (phase) {
      case 0:
        return ["references", "domain", "competitors"];
      case 1:
        return ["technical", "models", "examples"];
      case 5:
        return ["technical", "examples", "datasets"];
      case 6:
        return ["datasets", "validation"];
      default:
        return [];
    }
  }
};
function dedupe(arr) {
  return [...new Set(arr)];
}

// src/hooks/phase-gate.ts
function isAllowedPath(filePath) {
  const normalized = filePath.replace(/\\/g, "/").toLowerCase();
  const allowed = [
    "/specs/",
    "/.versus/",
    "/.claude/",
    "/package.json",
    "/tsconfig",
    "/readme",
    "/.gitignore",
    "/.mcp.json",
    "/.env"
  ];
  return allowed.some((p) => normalized.includes(p));
}
function deny(reason) {
  const output = {
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: reason
    }
  };
  process.stdout.write(JSON.stringify(output) + "\n");
  process.exit(0);
}
function allow() {
  process.exit(0);
}
function main() {
  const workspacePath = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const engine = new StateEngine(workspacePath);
  const state = engine.getPhaseState();
  if (!state) {
    allow();
    return;
  }
  if (state.currentPhase >= 5) {
    allow();
    return;
  }
  let input = "";
  process.stdin.setEncoding("utf-8");
  const timeoutId = setTimeout(() => {
    process.stdin.destroy();
    allow();
  }, 5e3);
  process.stdin.on("data", (chunk) => {
    input += chunk;
  });
  process.stdin.on("end", () => {
    clearTimeout(timeoutId);
    try {
      const hookInput = JSON.parse(input);
      const filePath = hookInput.tool_input?.file_path || "";
      if (isAllowedPath(filePath)) {
        allow();
      } else {
        deny(
          `[Versus] Blocked: Phase ${state.currentPhase} does not allow code editing. File: ${filePath}. Complete Phases 0\u21924 before implementing. Use get_exit_criteria() to see what is missing.`
        );
      }
    } catch {
      allow();
    }
  });
}
main();
