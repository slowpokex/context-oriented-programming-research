# Playwright Selectors Guide

This guide explains how to write reliable selectors in Playwright tests.

## Selector Priority

Use selectors in this order of preference:

### 1. getByRole() - Highest Priority

Most accessible and stable. Works with ARIA roles and accessible names.

```typescript
// Buttons
await page.getByRole('button', { name: 'Submit' }).click();
await page.getByRole('button', { name: /Sign in/i }).click();

// Links
await page.getByRole('link', { name: 'Home' }).click();

// Headings
await expect(page.getByRole('heading', { name: 'Welcome' })).toBeVisible();

// Form inputs
await page.getByRole('textbox', { name: 'Search' }).fill('query');
await page.getByRole('checkbox', { name: 'I agree' }).check();

// Navigation
await page.getByRole('navigation').getByRole('link', { name: 'About' }).click();
```

**When to use:**
- Buttons, links, form controls
- Elements with accessible names
- Navigation and landmarks

### 2. getByText() - For Visible Text

Good for finding elements by their visible text content.

```typescript
// Exact match
await page.getByText('Welcome back').click();

// Partial match (regex)
await page.getByText(/Welcome/).click();

// Case insensitive
await page.getByText(/welcome/i).click();
```

**When to use:**
- Text content that's visible to users
- Error messages, success messages
- Headings and labels

**Limitations:**
- Text can change (translations, content updates)
- May match multiple elements

### 3. getByLabel() - Form Inputs

Perfect for form inputs with associated labels.

```typescript
// By label text
await page.getByLabel('Email').fill('user@example.com');
await page.getByLabel('Password').fill('password123');

// By label with partial match
await page.getByLabel(/email address/i).fill('user@example.com');
```

**When to use:**
- Form inputs (text, email, password, etc.)
- Checkboxes and radio buttons
- Textareas

**Requirements:**
- Input must have a `<label>` element
- Or use `aria-label` attribute

### 4. getByPlaceholder() - Input Placeholders

For inputs that use placeholder text instead of labels.

```typescript
await page.getByPlaceholder('Enter your email').fill('user@example.com');
await page.getByPlaceholder(/search/i).fill('query');
```

**When to use:**
- Inputs with placeholder text
- Search boxes
- When label is not available

**Note:** Less reliable than `getByLabel()` - placeholders can change.

### 5. getByTestId() - Test IDs

Use when semantic selectors aren't available. Requires adding `data-testid` attributes.

```typescript
// In your HTML/JSX
<button data-testid="submit-button">Submit</button>

// In your test
await page.getByTestId('submit-button').click();
```

**When to use:**
- Complex components without semantic HTML
- When other selectors are too brittle
- For elements that change frequently

**Best practices:**
- Use descriptive names: `submit-button` not `btn1`
- Keep test IDs stable (don't change with styling)
- Don't overuse - prefer semantic selectors

### 6. CSS/XPath - Last Resort

Only use when no other selector works.

```typescript
// CSS selector
await page.locator('.submit-button').click();
await page.locator('#user-menu').click();

// XPath (avoid if possible)
await page.locator('xpath=//button[contains(text(), "Submit")]').click();
```

**When to use:**
- Legacy codebases without semantic HTML
- Complex component structures
- When other selectors fail

**Problems:**
- Brittle (breaks with styling changes)
- Hard to maintain
- Not accessible-friendly

## Selector Strategies

### Combining Selectors

```typescript
// Chain selectors for specificity
await page
  .getByRole('navigation')
  .getByRole('link', { name: 'Products' })
  .click();

// Filter by multiple criteria
await page
  .getByRole('button')
  .filter({ hasText: 'Delete' })
  .filter({ has: page.getByText('Confirm') })
  .click();
```

### Handling Multiple Matches

```typescript
// Get first match
await page.getByRole('button', { name: 'Submit' }).first().click();

// Get last match
await page.getByRole('button', { name: 'Submit' }).last().click();

// Get by index
await page.getByRole('button', { name: 'Submit' }).nth(2).click();

// Get all matches
const buttons = page.getByRole('button', { name: 'Submit' });
const count = await buttons.count();
```

### Filtering Selectors

```typescript
// Filter by text
await page.getByRole('button').filter({ hasText: 'Delete' }).click();

// Filter by visibility
await page.getByRole('button').filter({ hasNotText: 'Disabled' }).click();

// Filter by child element
await page
  .getByRole('listitem')
  .filter({ has: page.getByRole('button', { name: 'Edit' }) })
  .click();
```

## Common Patterns

### Form Inputs

```typescript
// Text input with label
await page.getByLabel('Username').fill('john_doe');

// Text input with placeholder
await page.getByPlaceholder('Enter username').fill('john_doe');

// Password input
await page.getByLabel('Password', { exact: true }).fill('secret123');

// Textarea
await page.getByLabel('Message').fill('Test message');

// Select dropdown
await page.getByLabel('Country').selectOption('United States');

// Checkbox
await page.getByLabel('I agree to terms').check();

// Radio button
await page.getByLabel('Option 1').check();
```

### Buttons and Links

```typescript
// Button by text
await page.getByRole('button', { name: 'Submit' }).click();

// Link by text
await page.getByRole('link', { name: 'Learn more' }).click();

// Icon button (use aria-label)
await page.getByRole('button', { name: 'Close dialog' }).click();
```

### Tables

```typescript
// Find row by text
await page.getByRole('row', { name: /John Doe/ }).click();

// Find cell
await page.getByRole('cell', { name: 'John Doe' }).click();

// Find in specific column
const table = page.getByRole('table');
await table.getByRole('row').filter({ hasText: 'John Doe' }).click();
```

### Modals and Dialogs

```typescript
// Find dialog
const dialog = page.getByRole('dialog');

// Find button in dialog
await dialog.getByRole('button', { name: 'Confirm' }).click();

// Find heading in dialog
await expect(dialog.getByRole('heading', { name: 'Confirm Action' })).toBeVisible();
```

## Anti-patterns

### ❌ Don't Use These

```typescript
// ❌ Brittle CSS classes
await page.locator('.btn-primary').click();

// ❌ Complex XPath
await page.locator('xpath=//div[@class="container"]//button[1]').click();

// ❌ Position-based selectors
await page.locator('button:nth-child(3)').click();

// ❌ Style-based selectors
await page.locator('[style*="color: red"]').click();

// ❌ Overly specific selectors
await page.locator('div.container > div.row > div.col-md-6 > button.btn').click();
```

### ✅ Use These Instead

```typescript
// ✅ Semantic role
await page.getByRole('button', { name: 'Submit' }).click();

// ✅ Test ID (if needed)
await page.getByTestId('submit-button').click();

// ✅ Accessible name
await page.getByLabel('Email').fill('user@example.com');
```

## Best Practices

1. **Prefer semantic selectors** - Use roles, labels, and accessible names
2. **Be specific but not brittle** - Target the right element without depending on structure
3. **Use test IDs sparingly** - Only when semantic selectors aren't available
4. **Avoid CSS classes** - They change with styling
5. **Document unusual selectors** - Explain why a non-standard selector was used
6. **Test accessibility** - Good selectors improve accessibility
7. **Keep selectors simple** - Complex selectors are harder to maintain

## Troubleshooting

### "Element not found"

```typescript
// Wait for element
await page.waitForSelector('[data-testid="element"]');

// Check if element exists
const exists = await page.getByTestId('element').count() > 0;

// Use more specific selector
await page.getByRole('button', { name: 'Submit', exact: true }).click();
```

### "Multiple elements found"

```typescript
// Be more specific
await page
  .getByRole('region', { name: 'Form' })
  .getByRole('button', { name: 'Submit' })
  .click();

// Use first/last/nth
await page.getByRole('button', { name: 'Submit' }).first().click();
```

### "Element not visible"

```typescript
// Wait for visibility
await page.getByRole('button', { name: 'Submit' }).waitFor({ state: 'visible' });

// Check visibility
const isVisible = await page.getByRole('button', { name: 'Submit' }).isVisible();
```
