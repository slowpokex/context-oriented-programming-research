# COP Build Process: Internal Implementation Details

## Overview

This document provides a comprehensive technical deep-dive into how the COP build process works under the hood. It covers data structures, algorithms, internal representations, and implementation strategies for each stage of the build pipeline.

---

## Table of Contents

1. [Internal Data Structures](#1-internal-data-structures)
2. [Stage 1: Load & Parse](#2-stage-1-load--parse)
3. [Stage 2: Dependency Resolution](#3-stage-2-dependency-resolution)
4. [Stage 3: Context Compilation](#4-stage-3-context-compilation)
5. [Stage 4: Validation](#5-stage-4-validation)
6. [Stage 5: Evaluation](#6-stage-5-evaluation)
7. [Stage 6: Optimization](#7-stage-6-optimization)
8. [Stage 7: Target Transformation](#8-stage-7-target-transformation)
9. [Stage 8: Artifact Generation](#9-stage-8-artifact-generation)
10. [Performance Optimizations](#10-performance-optimizations)
11. [Error Handling](#11-error-handling)
12. [Caching Strategy](#12-caching-strategy)

---

## 1. Internal Data Structures

### 1.1 Context AST (Abstract Syntax Tree)

The build process uses an AST to represent the entire context structure:

```typescript
interface ContextAST {
  meta: PackageMetadata;
  system: SystemPromptNode;
  personas: Map<string, PersonaNode>;
  guardrails: GuardrailNode[];
  knowledge: KnowledgeNode[];
  tools: ToolNode[];
  dependencies: DependencyNode[];
  variables: VariableBindings;
  buildConfig: BuildConfiguration;
}

interface SystemPromptNode {
  source: string;              // File path
  content: string;             // Raw template content
  compiled: string;            // After variable resolution
  variables: VariableDefinition[];
  includes: IncludeNode[];     // {{> includes}}
  conditionals: ConditionalNode[]; // {{#if}}
  loops: LoopNode[];           // {{#each}}
}

interface PersonaNode {
  name: string;
  source: string;
  content: YAMLObject;
  modifiers: PersonaModifier[];
  priority: number;
}

interface GuardrailNode {
  name: string;
  source: string;
  priority: number;
  rules: GuardrailRule[];
  triggers: TriggerPattern[];
  responses: ViolationResponse[];
  conflicts: ConflictMarker[]; // Detected conflicts
}

interface KnowledgeNode {
  name: string;
  source: string;
  type: 'static' | 'structured' | 'dynamic';
  content: string | JSONObject;
  schema?: JSONSchema;
  embedding?: VectorEmbedding; // For RAG
  chunks?: ChunkNode[];        // For large documents
}

interface ToolNode {
  name: string;
  source: string;
  definition: ToolDefinition;
  schema: JSONSchema;
  examples: ExamplePair[];
}
```

### 1.2 Dependency Graph

```typescript
interface DependencyGraph {
  nodes: Map<string, DependencyNode>;
  edges: DependencyEdge[];
  conflicts: Conflict[];
  resolution: ResolutionResult;
}

interface DependencyNode {
  name: string;
  version: string;
  constraint: VersionConstraint;
  resolvedVersion: string | null;
  dependencies: string[];      // Package names
  location: string;            // Registry URL or local path
  integrity: string;           // SHA256 hash
  evalFingerprints: Map<string, EvalFingerprint>;
}

interface DependencyEdge {
  from: string;
  to: string;
  type: 'direct' | 'transitive' | 'peer' | 'optional';
  constraint: VersionConstraint;
}
```

### 1.3 Build State

```typescript
interface BuildState {
  stage: BuildStage;
  contextAST: ContextAST;
  dependencyGraph: DependencyGraph;
  compiledContext: CompiledContext;
  validationResults: ValidationResults;
  evaluationResults: EvaluationResults;
  optimizedContext: OptimizedContext;
  targetArtifacts: Map<string, TargetArtifact>;
  errors: BuildError[];
  warnings: BuildWarning[];
  metrics: BuildMetrics;
}

enum BuildStage {
  LOADING = 'loading',
  RESOLVING = 'resolving',
  COMPILING = 'compiling',
  VALIDATING = 'validating',
  EVALUATING = 'evaluating',
  OPTIMIZING = 'optimizing',
  TRANSFORMING = 'transforming',
  PACKAGING = 'packaging',
  COMPLETE = 'complete',
  FAILED = 'failed'
}
```

---

## 2. Stage 1: Load & Parse

### 2.1 File Loading Algorithm

```typescript
async function loadContext(manifestPath: string): Promise<ContextAST> {
  // 1. Load and parse cop.yaml
  const manifest = await parseYAML(manifestPath);
  validateManifestSchema(manifest);
  
  // 2. Build file dependency graph
  const fileGraph = buildFileGraph(manifest);
  
  // 3. Load files in dependency order
  const loadedFiles = new Map<string, LoadedFile>();
  
  for (const file of topologicalSort(fileGraph)) {
    const content = await loadFile(file.path);
    const parsed = parseFile(file.path, content, file.type);
    loadedFiles.set(file.path, {
      content,
      parsed,
      hash: sha256(content),
      dependencies: file.dependencies
    });
  }
  
  // 4. Build AST
  return buildAST(manifest, loadedFiles);
}
```

### 2.2 Template Parsing

Templates use a Handlebars-like syntax:

```typescript
interface TemplateAST {
  type: 'text' | 'variable' | 'conditional' | 'loop' | 'include';
  content: string;
  children?: TemplateAST[];
  attributes?: Map<string, any>;
}

function parseTemplate(content: string): TemplateAST {
  const tokens = tokenize(content);
  const ast = parseTokens(tokens);
  return ast;
}

function tokenize(content: string): Token[] {
  const tokens: Token[] = [];
  const regex = /\{\{([#\/>!]?)([^}]+)\}\}/g;
  let lastIndex = 0;
  let match;
  
  while ((match = regex.exec(content)) !== null) {
    // Text before token
    if (match.index > lastIndex) {
      tokens.push({
        type: 'text',
        content: content.substring(lastIndex, match.index)
      });
    }
    
    // Token
    const prefix = match[1];
    const expression = match[2].trim();
    
    if (prefix === '#') {
      tokens.push({ type: 'block_start', expression });
    } else if (prefix === '/') {
      tokens.push({ type: 'block_end', expression });
    } else if (prefix === '>') {
      tokens.push({ type: 'include', expression });
    } else {
      tokens.push({ type: 'variable', expression });
    }
    
    lastIndex = regex.lastIndex;
  }
  
  // Remaining text
  if (lastIndex < content.length) {
    tokens.push({
      type: 'text',
      content: content.substring(lastIndex)
    });
  }
  
  return tokens;
}
```

### 2.3 YAML Parsing with Schema Validation

```typescript
function parseYAMLWithSchema(path: string, schema: JSONSchema): any {
  const content = fs.readFileSync(path, 'utf-8');
  const parsed = yaml.parse(content);
  
  // Validate against JSON Schema
  const validator = new Ajv();
  const validate = validator.compile(schema);
  const valid = validate(parsed);
  
  if (!valid) {
    throw new ValidationError(
      `Schema validation failed for ${path}`,
      validate.errors
    );
  }
  
  return parsed;
}
```

---

## 3. Stage 2: Dependency Resolution

### 3.1 Version Constraint Parsing

```typescript
interface VersionConstraint {
  operator: 'exact' | 'caret' | 'tilde' | 'gte' | 'lte' | 'range';
  version: string;
  min?: string;
  max?: string;
}

function parseConstraint(constraint: string): VersionConstraint {
  if (constraint === '*') {
    return { operator: 'gte', version: '0.0.0' };
  }
  
  if (constraint.startsWith('^')) {
    const version = constraint.substring(1);
    const [major] = version.split('.');
    return {
      operator: 'caret',
      version,
      min: version,
      max: `${parseInt(major) + 1}.0.0`
    };
  }
  
  if (constraint.startsWith('~')) {
    const version = constraint.substring(1);
    const [major, minor] = version.split('.');
    return {
      operator: 'tilde',
      version,
      min: version,
      max: `${major}.${parseInt(minor) + 1}.0`
    };
  }
  
  if (constraint.includes(' - ')) {
    const [min, max] = constraint.split(' - ');
    return {
      operator: 'range',
      min: min.trim(),
      max: max.trim()
    };
  }
  
  if (constraint.startsWith('>=')) {
    return {
      operator: 'gte',
      version: constraint.substring(2)
    };
  }
  
  // Exact version
  return {
    operator: 'exact',
    version: constraint
  };
}
```

### 3.2 Dependency Resolution Algorithm

Uses a SAT solver approach (similar to npm/yarn):

```typescript
class DependencyResolver {
  private registry: RegistryClient;
  private cache: Map<string, PackageMetadata[]>;
  
  async resolve(
    rootDependencies: Map<string, string>
  ): Promise<DependencyGraph> {
    const graph = new DependencyGraph();
    const queue: Queue<ResolveTask> = new Queue();
    
    // Add root dependencies
    for (const [name, constraint] of rootDependencies) {
      queue.enqueue({
        package: name,
        constraint: parseConstraint(constraint),
        depth: 0,
        parent: null
      });
    }
    
    // BFS resolution
    while (!queue.isEmpty()) {
      const task = queue.dequeue();
      
      // Skip if already resolved
      if (graph.hasNode(task.package)) {
        continue;
      }
      
      // Fetch available versions
      const versions = await this.fetchVersions(task.package);
      
      // Find compatible version
      const compatible = this.findCompatibleVersion(
        versions,
        task.constraint
      );
      
      if (!compatible) {
        throw new ResolutionError(
          `No compatible version found for ${task.package}@${task.constraint}`
        );
      }
      
      // Add to graph
      const node = graph.addNode({
        name: task.package,
        version: compatible.version,
        constraint: task.constraint,
        resolvedVersion: compatible.version,
        location: compatible.location,
        integrity: compatible.integrity
      });
      
      // Add edge from parent
      if (task.parent) {
        graph.addEdge({
          from: task.parent,
          to: task.package,
          type: 'direct',
          constraint: task.constraint
        });
      }
      
      // Fetch package manifest
      const manifest = await this.fetchManifest(
        task.package,
        compatible.version
      );
      
      // Queue dependencies
      for (const [depName, depConstraint] of manifest.dependencies) {
        queue.enqueue({
          package: depName,
          constraint: parseConstraint(depConstraint),
          depth: task.depth + 1,
          parent: task.package
        });
      }
    }
    
    // Detect conflicts
    this.detectConflicts(graph);
    
    return graph;
  }
  
  private findCompatibleVersion(
    versions: PackageVersion[],
    constraint: VersionConstraint
  ): PackageVersion | null {
    const compatible = versions.filter(v => 
      this.satisfiesConstraint(v.version, constraint)
    );
    
    if (compatible.length === 0) {
      return null;
    }
    
    // Return highest version (semver sort)
    return compatible.sort(compareVersions).reverse()[0];
  }
  
  private satisfiesConstraint(
    version: string,
    constraint: VersionConstraint
  ): boolean {
    switch (constraint.operator) {
      case 'exact':
        return version === constraint.version;
      case 'caret':
        return satisfiesCaret(version, constraint);
      case 'tilde':
        return satisfiesTilde(version, constraint);
      case 'gte':
        return compareVersions(version, constraint.version) >= 0;
      case 'range':
        return compareVersions(version, constraint.min!) >= 0 &&
               compareVersions(version, constraint.max!) <= 0;
      default:
        return false;
    }
  }
}
```

### 3.3 Conflict Detection

```typescript
function detectConflicts(graph: DependencyGraph): Conflict[] {
  const conflicts: Conflict[] = [];
  const versionMap = new Map<string, Set<string>>();
  
  // Collect all versions for each package
  for (const node of graph.nodes.values()) {
    if (!versionMap.has(node.name)) {
      versionMap.set(node.name, new Set());
    }
    versionMap.get(node.name)!.add(node.resolvedVersion!);
  }
  
  // Find packages with multiple versions
  for (const [packageName, versions] of versionMap) {
    if (versions.size > 1) {
      const conflict: Conflict = {
        package: packageName,
        versions: Array.from(versions),
        paths: findConflictPaths(graph, packageName),
        severity: 'error'
      };
      conflicts.push(conflict);
    }
  }
  
  // Detect guardrail conflicts
  const guardrailConflicts = detectGuardrailConflicts(graph);
  conflicts.push(...guardrailConflicts);
  
  return conflicts;
}

function findConflictPaths(
  graph: DependencyGraph,
  packageName: string
): string[][] {
  const paths: string[][] = [];
  const nodes = Array.from(graph.nodes.values())
    .filter(n => n.name === packageName);
  
  for (const node of nodes) {
    const path = findPathToRoot(graph, node);
    paths.push(path);
  }
  
  return paths;
}
```

---

## 4. Stage 3: Context Compilation

### 4.1 Template Variable Resolution

```typescript
class TemplateCompiler {
  private variables: Map<string, any>;
  private includes: Map<string, string>;
  
  compile(template: TemplateAST, variables: Map<string, any>): string {
    this.variables = variables;
    return this.compileNode(template);
  }
  
  private compileNode(node: TemplateAST): string {
    switch (node.type) {
      case 'text':
        return node.content;
        
      case 'variable':
        return this.resolveVariable(node.content);
        
      case 'conditional':
        return this.compileConditional(node);
        
      case 'loop':
        return this.compileLoop(node);
        
      case 'include':
        return this.compileInclude(node);
        
      default:
        return '';
    }
  }
  
  private resolveVariable(expression: string): string {
    // Support default values: {{name|default}}
    const [varName, defaultValue] = expression.split('|').map(s => s.trim());
    
    const value = this.getVariable(varName);
    if (value === undefined || value === null) {
      return defaultValue || '';
    }
    
    return String(value);
  }
  
  private compileConditional(node: TemplateAST): string {
    const condition = this.evaluateCondition(node.attributes!.condition);
    
    if (condition) {
      return this.compileChildren(node.children!);
    }
    
    // Check for else block
    const elseBlock = node.children?.find(c => c.type === 'else');
    if (elseBlock) {
      return this.compileChildren(elseBlock.children!);
    }
    
    return '';
  }
  
  private compileLoop(node: TemplateAST): string {
    const array = this.getVariable(node.attributes!.array);
    if (!Array.isArray(array)) {
      return '';
    }
    
    let result = '';
    for (const item of array) {
      const itemContext = new Map(this.variables);
      itemContext.set('this', item);
      itemContext.set('index', array.indexOf(item));
      
      const originalVars = this.variables;
      this.variables = itemContext;
      result += this.compileChildren(node.children!);
      this.variables = originalVars;
    }
    
    return result;
  }
  
  private compileInclude(node: TemplateAST): string {
    const includePath = node.attributes!.path;
    const includedContent = this.includes.get(includePath);
    
    if (!includedContent) {
      throw new Error(`Include not found: ${includePath}`);
    }
    
    const includedAST = parseTemplate(includedContent);
    return this.compileNode(includedAST);
  }
}
```

### 4.2 System Prompt Merging

```typescript
function mergeSystemPrompts(
  prompts: SystemPromptNode[],
  strategy: MergeStrategy = 'append'
): SystemPromptNode {
  switch (strategy) {
    case 'append':
      return mergeAppend(prompts);
    case 'prepend':
      return mergePrepend(prompts);
    case 'replace':
      return prompts[prompts.length - 1];
    case 'priority':
      return mergeByPriority(prompts);
    default:
      return mergeAppend(prompts);
  }
}

function mergeAppend(prompts: SystemPromptNode[]): SystemPromptNode {
  const merged: SystemPromptNode = {
    source: 'merged',
    content: '',
    compiled: '',
    variables: [],
    includes: [],
    conditionals: [],
    loops: []
  };
  
  for (const prompt of prompts) {
    merged.content += prompt.content + '\n\n';
    merged.variables.push(...prompt.variables);
    merged.includes.push(...prompt.includes);
    merged.conditionals.push(...prompt.conditionals);
    merged.loops.push(...prompt.loops);
  }
  
  return merged;
}

function mergeByPriority(prompts: SystemPromptNode[]): SystemPromptNode {
  // Sort by priority (higher first)
  const sorted = prompts.sort((a, b) => 
    (b.priority || 0) - (a.priority || 0)
  );
  
  // Merge with priority ordering
  const merged = mergeAppend(sorted);
  
  // Add priority markers for debugging
  merged.content = sorted.map((p, i) => 
    `[Priority ${p.priority || 0}] ${p.content}`
  ).join('\n\n');
  
  return merged;
}
```

### 4.3 Guardrail Merging with Conflict Detection

```typescript
function mergeGuardrails(
  guardrails: GuardrailNode[]
): GuardrailNode[] {
  // Sort by priority (higher first)
  const sorted = guardrails.sort((a, b) => 
    (b.priority || 0) - (a.priority || 0)
  );
  
  const merged: GuardrailNode[] = [];
  const ruleMap = new Map<string, GuardrailRule>();
  
  for (const guardrail of sorted) {
    for (const rule of guardrail.rules) {
      const ruleKey = normalizeRule(rule.content);
      
      // Check for conflicts
      if (ruleMap.has(ruleKey)) {
        const existing = ruleMap.get(ruleKey)!;
        if (isConflicting(existing, rule)) {
          // Mark conflict
          guardrail.conflicts.push({
            rule: rule.content,
            conflictingWith: existing.content,
            severity: calculateConflictSeverity(existing, rule)
          });
          
          // Higher priority wins
          if (guardrail.priority > existing.priority) {
            ruleMap.set(ruleKey, rule);
          }
        }
      } else {
        ruleMap.set(ruleKey, rule);
      }
    }
    
    merged.push(guardrail);
  }
  
  return merged;
}

function isConflicting(
  rule1: GuardrailRule,
  rule2: GuardrailRule
): boolean {
  // Use semantic similarity to detect conflicts
  const similarity = calculateSemanticSimilarity(
    rule1.content,
    rule2.content
  );
  
  // If very similar but with opposite sentiment, it's a conflict
  if (similarity > 0.8) {
    const sentiment1 = analyzeSentiment(rule1.content);
    const sentiment2 = analyzeSentiment(rule2.content);
    
    return areOpposite(sentiment1, sentiment2);
  }
  
  // Check for explicit contradictions
  const contradictions = [
    ['never', 'always'],
    ['must', 'must not'],
    ['required', 'forbidden']
  ];
  
  for (const [word1, word2] of contradictions) {
    if (rule1.content.includes(word1) && rule2.content.includes(word2)) {
      return true;
    }
  }
  
  return false;
}
```

### 4.4 Persona Selection and Application

```typescript
function selectPersona(
  personas: Map<string, PersonaNode>,
  defaultName: string
): PersonaNode {
  const persona = personas.get(defaultName);
  
  if (!persona) {
    throw new Error(`Persona not found: ${defaultName}`);
  }
  
  return persona;
}

function applyPersona(
  systemPrompt: SystemPromptNode,
  persona: PersonaNode
): SystemPromptNode {
  const modified: SystemPromptNode = {
    ...systemPrompt,
    content: systemPrompt.content + '\n\n' + formatPersona(persona)
  };
  
  // Apply persona modifiers
  for (const modifier of persona.modifiers) {
    modified.content = applyModifier(modified.content, modifier);
  }
  
  return modified;
}

function formatPersona(persona: PersonaNode): string {
  let formatted = `## Persona: ${persona.name}\n\n`;
  
  if (persona.content.tone) {
    formatted += `Tone: ${persona.content.tone}\n`;
  }
  
  if (persona.content.style) {
    formatted += `Style: ${persona.content.style}\n`;
  }
  
  if (persona.content.characteristics) {
    formatted += `Characteristics:\n`;
    for (const char of persona.content.characteristics) {
      formatted += `- ${char}\n`;
    }
  }
  
  return formatted;
}
```

---

## 5. Stage 4: Validation

### 5.1 Variable Completeness Check

```typescript
function validateVariables(
  context: CompiledContext,
  variables: VariableBindings
): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];
  
  // Collect all required variables from templates
  const requiredVars = collectRequiredVariables(context);
  
  // Check each required variable
  for (const varName of requiredVars) {
    if (!variables.has(varName)) {
      const varDef = findVariableDefinition(context, varName);
      
      if (varDef?.required) {
        errors.push({
          type: 'missing_required_variable',
          variable: varName,
          message: `Required variable '${varName}' is not provided`
        });
      } else if (varDef && !varDef.default) {
        warnings.push({
          type: 'missing_optional_variable',
          variable: varName,
          message: `Optional variable '${varName}' is not provided and has no default`
        });
      }
    } else {
      // Validate type
      const value = variables.get(varName);
      const varDef = findVariableDefinition(context, varName);
      
      if (varDef && !validateType(value, varDef.type)) {
        errors.push({
          type: 'type_mismatch',
          variable: varName,
          expected: varDef.type,
          actual: typeof value,
          message: `Variable '${varName}' has wrong type`
        });
      }
      
      // Validate format (email, URL, etc.)
      if (varDef?.format && !validateFormat(value, varDef.format)) {
        errors.push({
          type: 'format_mismatch',
          variable: varName,
          format: varDef.format,
          message: `Variable '${varName}' does not match format '${varDef.format}'`
        });
      }
    }
  }
  
  return { errors, warnings };
}
```

### 5.2 Token Count Estimation

```typescript
class TokenEstimator {
  private tokenizer: Tokenizer;
  
  constructor(model: string) {
    this.tokenizer = getTokenizer(model);
  }
  
  estimate(context: CompiledContext): TokenEstimate {
    const estimates: TokenEstimate = {
      systemPrompt: 0,
      persona: 0,
      guardrails: 0,
      knowledge: 0,
      tools: 0,
      total: 0,
      breakdown: []
    };
    
    // Estimate system prompt
    estimates.systemPrompt = this.tokenizer.count(context.systemPrompt.compiled);
    estimates.breakdown.push({
      component: 'system_prompt',
      tokens: estimates.systemPrompt
    });
    
    // Estimate persona
    if (context.activePersona) {
      estimates.persona = this.tokenizer.count(
        formatPersona(context.activePersona)
      );
      estimates.breakdown.push({
        component: 'persona',
        tokens: estimates.persona
      });
    }
    
    // Estimate guardrails
    for (const guardrail of context.guardrails) {
      const tokens = this.tokenizer.count(
        guardrail.rules.map(r => r.content).join('\n')
      );
      estimates.guardrails += tokens;
    }
    estimates.breakdown.push({
      component: 'guardrails',
      tokens: estimates.guardrails
    });
    
    // Estimate knowledge
    for (const knowledge of context.knowledge) {
      if (knowledge.type === 'static') {
        const tokens = this.tokenizer.count(knowledge.content);
        estimates.knowledge += tokens;
      }
    }
    estimates.breakdown.push({
      component: 'knowledge',
      tokens: estimates.knowledge
    });
    
    // Estimate tools
    for (const tool of context.tools) {
      const tokens = this.tokenizer.count(
        JSON.stringify(tool.definition)
      );
      estimates.tools += tokens;
    }
    estimates.breakdown.push({
      component: 'tools',
      tokens: estimates.tools
    });
    
    // Total
    estimates.total = 
      estimates.systemPrompt +
      estimates.persona +
      estimates.guardrails +
      estimates.knowledge +
      estimates.tools;
    
    return estimates;
  }
  
  checkLimits(
    estimates: TokenEstimate,
    model: string
  ): ValidationResult {
    const limits = getModelLimits(model);
    const errors: ValidationError[] = [];
    const warnings: ValidationWarning[] = [];
    
    if (estimates.total > limits.contextWindow) {
      errors.push({
        type: 'token_limit_exceeded',
        message: `Total tokens (${estimates.total}) exceed context window (${limits.contextWindow})`,
        total: estimates.total,
        limit: limits.contextWindow,
        excess: estimates.total - limits.contextWindow
      });
    } else if (estimates.total > limits.contextWindow * 0.9) {
      warnings.push({
        type: 'token_limit_warning',
        message: `Total tokens (${estimates.total}) are close to context window limit (${limits.contextWindow})`,
        total: estimates.total,
        limit: limits.contextWindow,
        percentage: (estimates.total / limits.contextWindow) * 100
      });
    }
    
    return { errors, warnings };
  }
}
```

### 5.3 Schema Validation

```typescript
function validateSchemas(context: CompiledContext): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];
  
  // Validate tool schemas
  for (const tool of context.tools) {
    try {
      const schema = tool.schema;
      const validator = new Ajv();
      const validate = validator.compile(schema);
      
      // Validate against JSON Schema spec
      const metaSchema = getJSONSchemaMetaSchema();
      const metaValidator = new Ajv();
      const metaValidate = metaValidator.compile(metaSchema);
      
      if (!metaValidate(schema)) {
        errors.push({
          type: 'invalid_schema',
          tool: tool.name,
          message: `Tool '${tool.name}' has invalid JSON Schema`,
          errors: metaValidate.errors
        });
      }
    } catch (error) {
      errors.push({
        type: 'schema_validation_error',
        tool: tool.name,
        message: `Failed to validate schema for tool '${tool.name}': ${error.message}`
      });
    }
  }
  
  // Validate knowledge schemas
  for (const knowledge of context.knowledge) {
    if (knowledge.type === 'structured' && knowledge.schema) {
      try {
        const data = JSON.parse(knowledge.content);
        const validator = new Ajv();
        const validate = validator.compile(knowledge.schema);
        
        if (!validate(data)) {
          errors.push({
            type: 'knowledge_schema_mismatch',
            knowledge: knowledge.name,
            message: `Knowledge '${knowledge.name}' does not match its schema`,
            errors: validate.errors
          });
        }
      } catch (error) {
        errors.push({
          type: 'knowledge_parse_error',
          knowledge: knowledge.name,
          message: `Failed to parse knowledge '${knowledge.name}': ${error.message}`
        });
      }
    }
  }
  
  return { errors, warnings };
}
```

---

## 6. Stage 5: Evaluation

### 6.1 Test Suite Execution

```typescript
class EvaluationEngine {
  private llmClient: LLMClient;
  private judgeModel: string;
  
  async evaluate(
    context: CompiledContext,
    testSuites: TestSuite[]
  ): Promise<EvaluationResults> {
    const results: EvaluationResults = {
      suites: [],
      aggregate: {
        passed: 0,
        failed: 0,
        total: 0,
        score: 0
      },
      fingerprint: null
    };
    
    for (const suite of testSuites) {
      const suiteResults = await this.runTestSuite(context, suite);
      results.suites.push(suiteResults);
      
      results.aggregate.total += suiteResults.total;
      results.aggregate.passed += suiteResults.passed;
      results.aggregate.failed += suiteResults.failed;
    }
    
    // Calculate aggregate score
    results.aggregate.score = 
      results.aggregate.passed / results.aggregate.total;
    
    // Generate fingerprint
    results.fingerprint = this.generateFingerprint(results);
    
    return results;
  }
  
  private async runTestSuite(
    context: CompiledContext,
    suite: TestSuite
  ): Promise<SuiteResults> {
    const results: SuiteResults = {
      name: suite.name,
      type: suite.type,
      tests: [],
      passed: 0,
      failed: 0,
      total: 0,
      score: 0
    };
    
    for (const test of suite.tests) {
      const testResult = await this.runTest(context, test);
      results.tests.push(testResult);
      results.total++;
      
      if (testResult.passed) {
        results.passed++;
      } else {
        results.failed++;
      }
    }
    
    results.score = results.passed / results.total;
    return results;
  }
  
  private async runTest(
    context: CompiledContext,
    test: TestCase
  ): Promise<TestResult> {
    switch (test.type) {
      case 'deterministic':
        return this.runDeterministicTest(context, test);
      case 'llm-judged':
        return this.runLLMJudgeTest(context, test);
      case 'adversarial':
        return this.runAdversarialTest(context, test);
      case 'comparison':
        return this.runComparisonTest(context, test);
      default:
        throw new Error(`Unknown test type: ${test.type}`);
    }
  }
}
```

### 6.2 LLM-as-Judge Implementation

```typescript
async function runLLMJudgeTest(
  context: CompiledContext,
  test: TestCase
): Promise<TestResult> {
  // 1. Execute the context with test input
  const response = await executeContext(context, test.input);
  
  // 2. Prepare judge prompt
  const judgePrompt = buildJudgePrompt(test, response);
  
  // 3. Call judge model
  const judgeResponse = await this.llmClient.complete({
    model: test.judgeModel || this.judgeModel,
    messages: [
      { role: 'system', content: judgePrompt.system },
      { role: 'user', content: judgePrompt.user }
    ],
    temperature: 0.0, // Deterministic judging
    response_format: { type: 'json_object' }
  });
  
  // 4. Parse judge response
  const judgment = JSON.parse(judgeResponse.content);
  
  // 5. Evaluate against rubric
  const score = evaluateRubric(judgment, test.rubric);
  const passed = score >= (test.threshold || 0.8);
  
  return {
    test: test.name,
    type: 'llm-judged',
    passed,
    score,
    judgment,
    response,
    input: test.input
  };
}

function buildJudgePrompt(test: TestCase, response: string): JudgePrompt {
  return {
    system: `You are an expert evaluator. Evaluate the following response according to the rubric.`,
    user: `
Test Case: ${test.name}
Input: ${test.input}
Response: ${response}

Rubric:
${formatRubric(test.rubric)}

Evaluate the response and provide:
1. A score (0.0 to 1.0) for each rubric criterion
2. An overall score
3. Brief justification for each score

Return your evaluation as JSON:
{
  "scores": {
    "criterion1": 0.9,
    "criterion2": 0.85,
    ...
  },
  "overall": 0.875,
  "justification": "..."
}
    `
  };
}
```

### 6.3 Fingerprint Generation

```typescript
function generateFingerprint(
  results: EvaluationResults,
  context: CompiledContext
): EvaluationFingerprint {
  // Hash test cases
  const testHash = sha256(
    JSON.stringify(results.suites.map(s => ({
      name: s.name,
      tests: s.tests.map(t => ({
        name: t.test,
        input: t.input,
        type: t.type
      }))
    })))
  );
  
  // Hash results
  const resultHash = sha256(
    JSON.stringify({
      suites: results.suites.map(s => ({
        name: s.name,
        score: s.score,
        passed: s.passed,
        total: s.total
      })),
      aggregate: results.aggregate
    })
  );
  
  // Model versions
  const modelVersions = new Map<string, string>();
  for (const suite of results.suites) {
    for (const test of suite.tests) {
      if (test.model) {
        modelVersions.set(test.model, getModelVersion(test.model));
      }
    }
  }
  
  return {
    testHash,
    resultHash,
    modelVersions: Object.fromEntries(modelVersions),
    date: new Date().toISOString(),
    packageVersion: context.meta.version,
    scores: {
      aggregate: results.aggregate.score,
      suites: Object.fromEntries(
        results.suites.map(s => [s.name, s.score])
      )
    }
  };
}
```

---

## 7. Stage 6: Optimization

### 7.1 Prompt Minification

```typescript
function minifyPrompt(prompt: string): string {
  // Remove extra whitespace
  let minified = prompt
    .replace(/\n{3,}/g, '\n\n')  // Max 2 newlines
    .replace(/[ \t]+/g, ' ')     // Collapse spaces
    .trim();
  
  // Remove comments (if supported)
  minified = minified.replace(/<!--.*?-->/gs, '');
  
  // Remove empty sections
  minified = minified.replace(/^##\s+.*\n\n$/gm, '');
  
  return minified;
}
```

### 7.2 Knowledge Compression

```typescript
class KnowledgeCompressor {
  private llmClient: LLMClient;
  
  async compress(
    knowledge: KnowledgeNode,
    targetTokens: number
  ): Promise<KnowledgeNode> {
    if (knowledge.type !== 'static') {
      return knowledge; // Only compress static knowledge
    }
    
    const currentTokens = estimateTokens(knowledge.content);
    
    if (currentTokens <= targetTokens) {
      return knowledge; // Already within target
    }
    
    // Strategy 1: Summarization
    if (currentTokens / targetTokens > 2) {
      return await this.summarize(knowledge, targetTokens);
    }
    
    // Strategy 2: Chunking
    return this.chunk(knowledge, targetTokens);
  }
  
  private async summarize(
    knowledge: KnowledgeNode,
    targetTokens: number
  ): Promise<KnowledgeNode> {
    const prompt = `Summarize the following content to approximately ${targetTokens} tokens while preserving key information:\n\n${knowledge.content}`;
    
    const summary = await this.llmClient.complete({
      model: 'gpt-4',
      messages: [{ role: 'user', content: prompt }],
      max_tokens: targetTokens * 2 // Allow some buffer
    });
    
    return {
      ...knowledge,
      content: summary.content,
      compressed: true,
      originalLength: knowledge.content.length
    };
  }
  
  private chunk(
    knowledge: KnowledgeNode,
    maxChunkTokens: number
  ): KnowledgeNode {
    const chunks = splitIntoChunks(knowledge.content, maxChunkTokens);
    
    return {
      ...knowledge,
      chunks: chunks.map((content, i) => ({
        index: i,
        content,
        tokens: estimateTokens(content),
        embedding: null // Will be generated at runtime
      })),
      type: 'dynamic', // Convert to dynamic retrieval
      content: null // Original content moved to chunks
    };
  }
}
```

### 7.3 Token Usage Optimization

```typescript
function optimizeTokenUsage(context: CompiledContext): OptimizedContext {
  const optimized: OptimizedContext = { ...context };
  
  // 1. Reorder by importance
  optimized.systemPrompt = reorderByImportance(
    context.systemPrompt,
    ['role', 'instructions', 'guidelines', 'examples']
  );
  
  // 2. Remove redundant guardrails
  optimized.guardrails = removeRedundantGuardrails(context.guardrails);
  
  // 3. Optimize tool descriptions
  optimized.tools = context.tools.map(tool => ({
    ...tool,
    definition: {
      ...tool.definition,
      description: condenseDescription(tool.definition.description)
    }
  }));
  
  // 4. Compress knowledge
  optimized.knowledge = context.knowledge.map(k => {
    if (k.type === 'static' && estimateTokens(k.content) > 1000) {
      return compressKnowledge(k);
    }
    return k;
  });
  
  return optimized;
}
```

---

## 8. Stage 7: Target Transformation

### 8.1 OpenAI Assistant Format Transformation

```typescript
function transformToOpenAI(
  context: CompiledContext
): OpenAIArtifact {
  const assistant: OpenAIAssistant = {
    name: context.meta.name,
    instructions: context.systemPrompt.compiled,
    model: context.runtime.modelConfig?.model || 'gpt-4',
    tools: context.tools.map(tool => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.definition.description,
        parameters: tool.schema
      }
    })),
    tool_resources: {}
  };
  
  // Add knowledge files
  const knowledgeFiles: string[] = [];
  for (const knowledge of context.knowledge) {
    if (knowledge.type === 'static') {
      const fileId = uploadKnowledgeFile(knowledge.content);
      knowledgeFiles.push(fileId);
    }
  }
  
  if (knowledgeFiles.length > 0) {
    assistant.tool_resources = {
      file_search: {
        vector_store_ids: knowledgeFiles
      }
    };
  }
  
  return {
    format: 'openai-assistant',
    assistant,
    files: knowledgeFiles
  };
}
```

### 8.2 Anthropic Claude Format Transformation

```typescript
function transformToAnthropic(
  context: CompiledContext
): AnthropicArtifact {
  // Claude prefers XML-structured prompts
  const systemPrompt = formatAsXML(context);
  
  const tools = context.tools.map(tool => ({
    name: tool.name,
    description: tool.definition.description,
    input_schema: tool.schema
  }));
  
  return {
    format: 'anthropic-claude',
    system: systemPrompt,
    tools: tools.length > 0 ? tools : undefined,
    model: 'claude-3-opus-20240229'
  };
}

function formatAsXML(context: CompiledContext): string {
  let xml = '<system_prompt>\n';
  
  xml += '  <instructions>\n';
  xml += `    ${escapeXML(context.systemPrompt.compiled)}\n`;
  xml += '  </instructions>\n';
  
  if (context.activePersona) {
    xml += '  <persona>\n';
    xml += `    ${formatPersonaXML(context.activePersona)}\n`;
    xml += '  </persona>\n';
  }
  
  xml += '  <guardrails>\n';
  for (const guardrail of context.guardrails) {
    xml += `    <rule priority="${guardrail.priority}">\n`;
    for (const rule of guardrail.rules) {
      xml += `      ${escapeXML(rule.content)}\n`;
    }
    xml += '    </rule>\n';
  }
  xml += '  </guardrails>\n';
  
  xml += '</system_prompt>';
  return xml;
}
```

---

## 9. Stage 8: Artifact Generation

### 9.1 Package Archive Creation

```typescript
function createPackageArchive(
  artifacts: Map<string, TargetArtifact>,
  context: CompiledContext
): PackageArchive {
  const archive = new TarArchive();
  
  // Add manifest
  archive.addFile('cop.yaml', serializeYAML(context.meta));
  
  // Add lock file
  archive.addFile('cop.lock', serializeYAML(context.dependencyGraph));
  
  // Add source files
  archive.addDirectory('prompts', context.sourceFiles.prompts);
  archive.addDirectory('personas', context.sourceFiles.personas);
  archive.addDirectory('guardrails', context.sourceFiles.guardrails);
  archive.addDirectory('knowledge', context.sourceFiles.knowledge);
  archive.addDirectory('tools', context.sourceFiles.tools);
  
  // Add compiled artifacts
  archive.addDirectory('dist', {
    'context.bundle.json': JSON.stringify(context.compiled),
    'evaluation/fingerprint.json': JSON.stringify(context.evaluation.fingerprint)
  });
  
  // Add target-specific artifacts
  for (const [target, artifact] of artifacts) {
    archive.addDirectory(`dist/${target}`, artifact.files);
  }
  
  // Generate checksums
  const checksums = generateChecksums(archive);
  archive.addFile('checksums.sha256', checksums);
  
  return archive;
}
```

### 9.2 Checksum Generation

```typescript
function generateChecksums(archive: TarArchive): string {
  const checksums: string[] = [];
  
  for (const file of archive.files) {
    const hash = sha256(file.content);
    checksums.push(`${hash}  ${file.path}`);
  }
  
  return checksums.join('\n');
}
```

---

## 10. Performance Optimizations

### 10.1 Parallel Processing

```typescript
async function buildParallel(
  context: ContextAST
): Promise<BuildState> {
  // Stage 1 & 2 can run in parallel for dependencies
  const [loaded, dependencies] = await Promise.all([
    loadContext(context.manifestPath),
    resolveDependencies(context.dependencies)
  ]);
  
  // Stage 3: Compile templates in parallel
  const compiled = await Promise.all([
    compileSystemPrompt(loaded.system),
    compilePersonas(loaded.personas),
    compileGuardrails(loaded.guardrails),
    compileKnowledge(loaded.knowledge),
    compileTools(loaded.tools)
  ]);
  
  // Stage 5: Run test suites in parallel
  const evaluations = await Promise.all(
    context.testSuites.map(suite => runTestSuite(compiled, suite))
  );
  
  return { ...compiled, evaluations };
}
```

### 10.2 Incremental Builds

```typescript
class IncrementalBuilder {
  private cache: BuildCache;
  
  async build(
    context: ContextAST,
    force: boolean = false
  ): Promise<BuildState> {
    // Check cache
    const cacheKey = generateCacheKey(context);
    if (!force && this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey);
      if (cached.isValid(context)) {
        return cached.state;
      }
    }
    
    // Build only changed parts
    const changed = this.detectChanges(context);
    const state = await this.buildIncremental(context, changed);
    
    // Update cache
    this.cache.set(cacheKey, {
      state,
      timestamp: Date.now(),
      isValid: (ctx) => !this.detectChanges(ctx).length
    });
    
    return state;
  }
  
  private detectChanges(context: ContextAST): string[] {
    const changed: string[] = [];
    const previous = this.cache.getPrevious(context);
    
    if (!previous) {
      return ['all']; // First build
    }
    
    // Compare file hashes
    for (const file of context.files) {
      const currentHash = sha256(file.content);
      const previousHash = previous.fileHashes.get(file.path);
      
      if (currentHash !== previousHash) {
        changed.push(file.path);
      }
    }
    
    return changed;
  }
}
```

### 10.3 Caching Strategy

```typescript
class BuildCache {
  private cache: Map<string, CachedBuild>;
  private ttl: number = 3600000; // 1 hour
  
  has(key: string): boolean {
    const cached = this.cache.get(key);
    if (!cached) return false;
    
    // Check TTL
    if (Date.now() - cached.timestamp > this.ttl) {
      this.cache.delete(key);
      return false;
    }
    
    return true;
  }
  
  get(key: string): CachedBuild {
    return this.cache.get(key)!;
  }
  
  set(key: string, build: CachedBuild): void {
    this.cache.set(key, build);
    
    // Evict old entries if cache is too large
    if (this.cache.size > 1000) {
      this.evictOldest();
    }
  }
  
  private evictOldest(): void {
    const entries = Array.from(this.cache.entries())
      .sort((a, b) => a[1].timestamp - b[1].timestamp);
    
    // Remove oldest 10%
    const toRemove = Math.floor(entries.length * 0.1);
    for (let i = 0; i < toRemove; i++) {
      this.cache.delete(entries[i][0]);
    }
  }
}
```

---

## 11. Error Handling

### 11.1 Error Recovery

```typescript
class BuildErrorHandler {
  handleError(
    error: BuildError,
    state: BuildState
  ): ErrorRecovery {
    switch (error.type) {
      case 'missing_dependency':
        return this.suggestAlternative(error, state);
        
      case 'version_conflict':
        return this.suggestResolution(error, state);
        
      case 'validation_failure':
        return this.suggestFix(error, state);
        
      case 'evaluation_failure':
        return this.suggestImprovements(error, state);
        
      default:
        return { recoverable: false, message: error.message };
    }
  }
  
  private suggestAlternative(
    error: MissingDependencyError,
    state: BuildState
  ): ErrorRecovery {
    // Search registry for similar packages
    const alternatives = searchRegistry(error.packageName);
    
    return {
      recoverable: true,
      message: `Package '${error.packageName}' not found`,
      suggestions: [
        `Install alternative: ${alternatives[0]?.name}`,
        `Check if package name is correct`,
        `Verify registry connection`
      ],
      actions: [
        { type: 'install', package: alternatives[0]?.name }
      ]
    };
  }
}
```

---

## 12. Caching Strategy

### 12.1 Multi-Level Caching

```typescript
class MultiLevelCache {
  private l1: MemoryCache;      // In-memory, fast
  private l2: DiskCache;        // On disk, persistent
  private l3: RegistryCache;    // Remote, shared
  
  async get(key: string): Promise<CachedItem | null> {
    // L1: Memory
    const l1Item = this.l1.get(key);
    if (l1Item) return l1Item;
    
    // L2: Disk
    const l2Item = await this.l2.get(key);
    if (l2Item) {
      this.l1.set(key, l2Item); // Promote to L1
      return l2Item;
    }
    
    // L3: Registry
    const l3Item = await this.l3.get(key);
    if (l3Item) {
      await this.l2.set(key, l3Item); // Promote to L2
      this.l1.set(key, l3Item);       // Promote to L1
      return l3Item;
    }
    
    return null;
  }
  
  async set(key: string, item: CachedItem): Promise<void> {
    // Set in all levels
    this.l1.set(key, item);
    await this.l2.set(key, item);
    await this.l3.set(key, item);
  }
}
```

---

## Conclusion

This document provides a comprehensive technical deep-dive into the internal implementation of the COP build process. The build system uses:

- **AST-based representation** for context structure
- **SAT solver algorithms** for dependency resolution
- **Template compilation** with variable resolution
- **Priority-based merging** for guardrails and prompts
- **LLM-as-judge** for behavioral evaluation
- **Multi-target transformation** for different providers
- **Incremental builds** with caching for performance

The implementation is designed to be:
- **Modular**: Each stage is independent and testable
- **Performant**: Parallel processing and caching
- **Robust**: Comprehensive error handling and recovery
- **Extensible**: Easy to add new targets and optimizations

---

*Document Version: 1.0.0*  
*Last Updated: December 2025*
