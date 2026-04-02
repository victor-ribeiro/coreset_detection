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
var PHASE_DEFINITIONS = [
  {
    number: 0,
    name: "Problem Discovery",
    description: "Iterative problem discovery loop. Score >= 90/100 to advance.",
    userHint: "Explore the problem thoroughly. No coding yet.",
    iterative: true
  },
  {
    number: 1,
    name: "Architecture",
    description: "Module, interface, pattern, and principle definitions.",
    userHint: "Design modules and interfaces. No coding yet.",
    iterative: false
  },
  {
    number: 2,
    name: "Adversarial Critique",
    description: "Attack architecture with specialized lenses. Generates coverage matrix.",
    userHint: "Challenge the architecture with adversarial lenses.",
    iterative: true
  },
  {
    number: 3,
    name: "Simplification",
    description: "Simplify architecture. Address Phase 2 critical findings.",
    userHint: "Simplify \u2014 address criticals without adding features.",
    iterative: true
  },
  {
    number: 4,
    name: "Convergence Gate",
    description: "Validate convergence: exit criteria from Phases 2-3 + safeguards S1-S5.",
    userHint: "Verify everything before implementation begins.",
    iterative: false
  },
  {
    number: 5,
    name: "Code Implementation",
    description: "Implement final architecture. One file per response.",
    userHint: "Implement code. Testing happens in Phase 6.",
    iterative: false
  },
  {
    number: 6,
    name: "Tests",
    description: "100% tests passing + manual exploratory testing.",
    userHint: "Write and run tests. Manual testing is mandatory.",
    iterative: false
  },
  {
    number: 7,
    name: "Post-Review",
    description: "Lessons learned, update specs/, methodology meta-iteration.",
    userHint: "Review lessons. Offer meta-iteration for next cycle.",
    iterative: false
  }
];
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
function getPhaseDefinition(phase) {
  return PHASE_DEFINITIONS.find((p) => p.number === phase);
}
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

// src/rules/compact-guidance.ts
function getCompactGuidance(phase) {
  const lines = [];
  switch (phase) {
    case 0:
      lines.push("INSTRUCTIONS P0: FIRST ask Delivery Target (Complete product / MVP / Prototype) via AskUserQuestion \u2014 record_decision(category='scope', content='DELIVERY TARGET: ...'). THEN HSA 5-level exploration (1.Domain 2.Problem-5W1H 3.Elements 4.Processes 5.Product). AskUserQuestion per level. Mandatory teach-back. Technical+scientific research if specialized domain. Tier 1/2/3 feasibility. Score \u226590.");
      lines.push("R2: Every decision via AskUserQuestion. AP1: GENERATION mode \u2014 produce failure scenarios ('how does this fail?'), not quality judgments ('is this good?').");
      lines.push("R5: MUST DISPLAY score breakdown table + teach-back synthesis + delivery target question. If you computed it, SHOW it.");
      lines.push("FLOW: After recording user answers \u2192 proceed IMMEDIATELY to next HSA level or to synthesis. NEVER write 'I will proceed with X' as text and stop \u2014 just do X.");
      lines.push("AFTER SCORE: score \u226590 \u2192 advance_phase() to Phase 1, then IMMEDIATELY call get_phase_guidance() and begin Phase 1 work \u2014 do NOT stop. Score <90 \u2192 call start_iteration() and continue HSA focusing on lowest-scoring criteria. Do NOT offer to advance.");
      lines.push("DO: Ask questions, explore the problem, populate specs/. | DON'T: Write source code (even via terminal), jump to solutions, skip HSA levels, stop after displaying score.");
      break;
    case 1:
      lines.push("INSTRUCTIONS P1: 4 questions (Decomposition, Interfaces, Assumptions, Negative scope). 2 AskUserQuestion calls (patterns). Granularity: each module fits in 1 AI session. UI MODULE: if project requires UI, list it as an explicit named module in decomposition with its own interface contract \u2014 never as an implied extension of backend.");
      lines.push("FLOW: After recording user answers \u2192 proceed IMMEDIATELY to draft the 4 architecture answers (decomposition, interfaces, assumptions, negative scope) and present the pattern summary table. NEVER write 'I will draft X next' as text and stop \u2014 just do it.");
      lines.push("R2+R4: specs/technical,models,examples. AP3: plan decoupled sessions for P5-P6. R5: MUST DISPLAY 4 architecture answers + pattern table + tech comparison.");
      lines.push("WHEN DONE: After user confirms architecture \u2192 call update_project_spec(stack=[approved techs], outOfScope=[items NOT done], patterns=[architectural patterns], constraints=[hard constraints]), then advance_phase(), then get_phase_guidance() and begin Phase 2 immediately.");
      lines.push("DO: Define modules, interfaces, patterns via AskUserQuestion. | DON'T: Write implementation code, create source files, start coding.");
      break;
    case 2:
      lines.push("INSTRUCTIONS P2: STEP 1 \u2014 Record activated lenses FIRST: call record_decision(category='architecture', content='ACTIVATED LENSES: [list]. NOT ACTIVATED: [lens \u2014 reason]') before starting critique. STEP 2 \u2014 Apply ALL 7 UNIVERSAL lenses on EACH module \u2192 \u{1F534}\u{1F7E1}\u{1F7E2}: (1)Assumptions (2)Architectural (3)Implementability (4)Scientific (5)Security (6)Performance (7)Regulatory. Then apply all activated conditional lenses. Absence of findings is valid \u2014 do NOT skip any lens. Matrix + concentration analysis (module\xD7lenses). ASYMMETRY: multiple agents attack, unified responds in P3.");
      lines.push("MODULE BOUNDARY: Apply lenses to ONLY the modules defined in P1 \u2014 do NOT propose new module boundaries or restructure decomposition here. If module granularity is insufficient, flag it as a finding for P3 to resolve.");
      lines.push("FLOW: After recording activated lenses \u2192 immediately start applying them (lens 1 on all modules \u2192 lens 2 \u2192 ... \u2192 exit gate). NEVER write 'I will now apply lens X' as text and stop \u2014 just apply it.");
      lines.push("BOUNDARY: do NOT cut/add requirements. P1 negative scope items CANNOT become implementation proposals \u2014 flag as out-of-scope if they appear. AP1: GENERATION mode \u2014 'how does this fail?' (generative, no bias anchor), not 'is this good?' (evaluative, confirms your own output). R5: MUST DISPLAY full coverage matrix + concentration analysis + exit gate summary.");
      lines.push("AFTER GATE: If findings \u2192 YOUR NEXT TOOL CALL is advance_phase() to Phase 3 \u2014 do NOT write 'advancing' as text, then get_phase_guidance() and begin Phase 3. If zero findings \u2192 AskUserQuestion (Phase 3 optional or Phase 4). Do NOT block on \u{1F534} count \u2014 criticals are addressed IN Phase 3.");
      lines.push("DO: Record lenses, apply all, advance to Phase 3 after gate. | DON'T: Block advancement because of \u{1F534} criticals, stop silently after gate summary, skip any lens.");
      break;
    case 3:
      lines.push("INSTRUCTIONS P3: Simplify V(N)\u2192V(N+1) with UNIFIED vision. AI COMPUTES structural change % and critical count, then presents recommendation. User confirms or overrides. Anti-scope-creep checklist.");
      lines.push("FLOW: Immediately compute change % and critical count, then display recommendation and checklist \u2014 all in one continuous turn. NEVER write 'I will compute X' as text and stop \u2014 just compute and display.");
      lines.push("ASYMMETRY: integrate all corrections into the whole. AP2: simplify, not complexify. AP8: don't seek perfection. R5: MUST DISPLAY computed % + criticals + recommendation + checklist \u2705/\u274C. Do NOT ask user to choose percentage range.");
      lines.push("AFTER CONFIRMATION: YOUR NEXT TOOL CALL \u2014 '\u2192 Phase 4': advance_phase() then get_phase_guidance() and begin P4. 'More iterations': start_iteration() then advance_phase() to Phase 2, then get_phase_guidance(). Do NOT write 'advancing' as text \u2014 call the tool immediately.");
      lines.push("ANTI-SCOPE-CREEP checklist (MUST DISPLAY \u2705/\u274C for each): 1. All P0 use cases still covered. 2. No unsolicited features added. 3. WHAT the system does hasn't changed (only HOW). 4. P1 negative scope is still negative \u2014 no item from P1 'NOT do' list was silently promoted.");
      lines.push("DO: Simplify architecture, address criticals, compute change %, verify P1 negative scope. | DON'T: Add features, increase complexity, stop after confirmation without executing the action.");
      break;
    case 4:
      lines.push("INSTRUCTIONS P4: Convergence gate. Exit criteria P2-P3 + safeguards S1-S5 + 4 P1 questions + specs/ complete. get_exit_criteria() + check_all_safeguards().");
      lines.push("FLOW: Call get_exit_criteria() then check_all_safeguards() then display convergence report \u2014 all in one continuous turn. NEVER write 'I will now call X' as text and stop \u2014 just call it.");
      lines.push("WHEN ALL \u2705: call advance_phase() to Phase 5, then get_phase_guidance() and begin Phase 5 immediately. R5: MUST DISPLAY convergence report \u2705/\u274C. LLM Switch notification is in full guidance (get_phase_guidance()) \u2014 do NOT display it again in this turn.");
      lines.push("DO: Verify all exit criteria, run safeguard checks, check specs/. | DON'T: Skip verification steps, advance without confirming convergence, display LLM Switch notification twice.");
      break;
    case 5:
      lines.push("INSTRUCTIONS P5: Consult specs/ BEFORE each module (specs/technical, specs/examples, specs/models). S6: Tier 1(lib)\u21922(port)\u21923(new). S7: After each module, run ADVERSARIAL MICRO-CHECK: 'Where does this implementation DIVERGE from spec?' \u2014 find differences, do NOT validate. Fix any divergence before the next module. Do NOT write text between modules \u2014 YOUR NEXT TOOL CALL must be the first tool call of module X+1.");
      lines.push("FLOW: [per module: consult specs/ \u2192 implement \u2192 micro-check divergence \u2192 fix if needed \u2192 next] \u2192 scope inventory \u2192 advance. NEVER write text between modules ('X/N completed', 'I will now implement X', 'Next:') \u2014 just execute the next tool call.");
      lines.push("BINDING CONSTRAINTS: The tech stack, patterns, and out-of-scope items decided in P1/P3 are HARD CONSTRAINTS \u2014 visible in the 'Decision history' above. DO NOT introduce any library, framework, runtime, or technology not explicitly selected in P1. If the decision says 'vanilla JS', do not use Node.js. If it says 'no framework', do not use one.");
      lines.push("Do NOT ask permission between files \u2014 implement autonomously. AP7: never code without reference. STOP only for blockers or >2 iter on same module.");
      lines.push("WHEN ALL MODULES DONE: run SCOPE INVENTORY (\u26A0\uFE0F MUST DISPLAY \u2705/\u274C): 1. All P1 modules present in codebase? 2. All specs/validation requirements covered? 3. UI runnable? If UI module in P1: implemented + accessible for manual testing \u2192 mark_exit_criteria(phase=5, criterion='ui_runnable', met=true). If no UI \u2192 mark_exit_criteria(phase=5, criterion='ui_runnable', met=true, details='N/A\u2014backend-only'). Fix any \u274C before advancing. WHEN SCOPE INVENTORY ALL \u2705: do NOT end your response after the inventory table \u2014 your VERY NEXT action must be a tool call to advance_phase() (no text between the table and the tool call). Then call get_phase_guidance() and begin Phase 6. Never return control to the user between inventory and advance_phase().");
      lines.push("DO: Implement modules sequentially, micro-check divergence after each, run scope inventory before advancing. | DON'T: Validate instead of seeking divergence, introduce unapproved tech, run tests \u2014 that is Phase 6.");
      break;
    case 6:
      lines.push("INSTRUCTIONS P6: SPEC-DRIVEN testing \u2014 test against SPECS, not implementation. Step 1: Review Decision history above (P0 use cases + P3 scope revisions) + read specs/validation + specs/technical. P3 may have revised scope boundaries \u2014 merge both. Step 2: build Test Map (\u26A0\uFE0F MUST DISPLAY) mapping every spec\u2192test with Type(positive/negative). Every use case: \u22651 positive + \u22651 negative. Ratio: 1 negative per 2 positives minimum.");
      lines.push("FLOW: Recover scope \u2192 build Test Map \u2192 execute automated tests \u2192 record coverage \u2192 hand off to user \u2014 all without stopping to announce next steps. NEVER write 'I will now test X' as text and stop \u2014 just do it.");
      lines.push("DISTINGUISH 'test green' vs 'spec met': test must verify EXACT criterion (not proxy). Step 3: record_decision(category='spec-coverage') with COVERED/NOT COVERED/NEGATIVE TESTS/GAPS. Step 4: Manual testing \u2014 HUMAN USER tests, NOT AI. Present test plan, record_decision(category='testing') with STATUS/PASS/FAIL/NEXT, then WAIT for user to return. Only mark manual_testing met after user confirms.");
      lines.push("WHEN TESTING COMPLETE: YOUR NEXT TOOL CALL is advance_phase() to Phase 7, then get_phase_guidance() and begin Phase 7 immediately.");
      lines.push("BINDING CONSTRAINTS in test code: use ONLY the tech stack approved in P1 \u2014 visible in Decision history above. Do NOT introduce test frameworks or runtimes not approved in P1.");
      lines.push("DO: Write spec-driven tests, execute them, hand off manual testing to user, wait for return. | DON'T: Simulate manual testing mentally, mark manual_testing without user confirmation, use unapproved tech in test code.");
      break;
    case 7:
      lines.push("INSTRUCTIONS P7: Double-loop: evaluate PRODUCT and PROCESS. Update specs/. 5 steps: human first\u2192AI\u2192consolidate\u2192apply\u2192version.");
      lines.push("Meta-iteration: After completing P7, offer start_new_cycle() via AskUserQuestion. NEVER use init_project() for v2 \u2014 it wipes context. NEVER modify SKILL.md without approval. R5: MUST DISPLAY double-loop report + consolidation table + meta-iteration offer.");
      lines.push("DO: Collect human feedback first, update specs/, offer meta-iteration. | DON'T: Use init_project() instead of start_new_cycle(), modify SKILL.md without approval.");
      break;
  }
  return lines;
}

// src/hooks/inject-context.ts
function main() {
  const workspacePath = process.env.CLAUDE_PROJECT_DIR || process.cwd();
  const engine = new StateEngine(workspacePath);
  const state = engine.getPhaseState();
  if (!state) {
    process.exit(0);
  }
  const phase = state.currentPhase;
  const phaseName = getPhaseDefinition(phase)?.name ?? "Unknown";
  const lines = [
    `[Versus] Phase ${phase} \u2014 ${phaseName} | Iteration ${state.currentIteration}`
  ];
  if (state.phase0Score !== null) {
    lines.push(`Score Phase 0: ${state.phase0Score}/100`);
  }
  if (state.decisions.length > 0) {
    const byPhase = {};
    for (const d of state.decisions) {
      if (!byPhase[d.phase]) byPhase[d.phase] = [];
      byPhase[d.phase].push(d);
    }
    lines.push("Decision history (all phases):");
    for (const p of Object.keys(byPhase).map(Number).sort((a, b) => a - b)) {
      for (const d of byPhase[p]) {
        lines.push(`  [P${d.phase}/${d.category}] ${d.content}`);
      }
    }
  }
  const spec = state.projectSpec;
  if (spec) {
    if (spec.stack.length > 0) lines.push(`Approved stack: ${spec.stack.join(", ")}`);
    if (spec.outOfScope.length > 0) lines.push(`Out of scope: ${spec.outOfScope.join(", ")}`);
    if (spec.patterns.length > 0) lines.push(`Patterns: ${spec.patterns.join(", ")}`);
    if (spec.constraints.length > 0) lines.push(`Constraints: ${spec.constraints.join(", ")}`);
  }
  if (phase === 2) {
    const hasLensDecision = state.decisions.some((d) => d.content.includes("ACTIVATED LENSES"));
    if (!hasLensDecision) {
      lines.push("\u26A0 P2 STEP 1 MISSING: No lens activation decision found. Call record_decision(category='architecture', content='ACTIVATED LENSES: [list]. NOT ACTIVATED: [lens \u2014 reason]') BEFORE applying any lens.");
    }
  }
  if (phase < 5) {
    lines.push("RESTRICTION: Phases 0-4 \u2014 DO NOT implement code. Use the MCP tools to manage the methodology.");
  }
  const compactGuidance = getCompactGuidance(phase);
  for (const line of compactGuidance) {
    lines.push(line);
  }
  const loop = engine.getLoopCounter();
  if (loop.count >= 2) {
    lines.push(`\u26A0 Loop detector: pattern "${loop.pattern}" executed ${loop.count}x. Threshold: 3.`);
  }
  process.stdout.write(lines.join("\n") + "\n");
  process.exit(0);
}
main();
