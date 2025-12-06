# Playwright Testing Best Practices

This document outlines best practices for writing Playwright end-to-end tests.

## Selector Strategy

### Preferred Selectors (in order)

1. **getByRole()** - Most accessible and stable
   ```typescript
   await page.getByRole('button', { name: 'Submit' }).click();
   ```

2. **getByText()** - Good for visible text
   ```typescript
   await page.getByText('Welcome back').click();
   ```

3. **getByLabel()** - Perfect for form inputs
   ```typescript
   await page.getByLabel('Email').fill('user@example.com');
   ```

4. **getByPlaceholder()** - For inputs with placeholders
   ```typescript
   await page.getByPlaceholder('Enter your email').fill('user@example.com');
   ```

5. **getByTestId()** - When semantic selectors aren't available
   ```typescript
   await page.getByTestId('submit-button').click();
   ```

6. **CSS/XPath** - Last resort only
   ```typescript
   // Avoid if possible
   await page.locator('.submit-btn').click();
   ```

### Selector Anti-patterns

- ❌ CSS classes that change with styling
- ❌ Complex XPath expressions
- ❌ Selectors that depend on DOM structure
- ❌ Selectors that target multiple elements

## Waiting Strategies

### Playwright Auto-Waiting

Playwright automatically waits for elements to be:
- Attached to the DOM
- Visible
- Stable (not animating)
- Enabled
- Receiving events

**Don't add unnecessary waits:**

```typescript
// ❌ Bad
await page.waitForTimeout(5000);
await page.click('.button');

// ✅ Good - Playwright waits automatically
await page.getByRole('button').click();
```

### When to Add Explicit Waits

1. **Network Requests**
   ```typescript
   await page.waitForResponse(response => 
     response.url().includes('/api/login') && response.status() === 200
   );
   ```

2. **Page Load States**
   ```typescript
   await page.waitForLoadState('networkidle');
   ```

3. **Custom Conditions**
   ```typescript
   await page.waitForSelector('[data-testid="success-message"]', {
     state: 'visible'
   });
   ```

## Test Organization

### Structure

```typescript
import { test, expect } from '@playwright/test';

test.describe('Feature Name', () => {
  test.beforeEach(async ({ page }) => {
    // Setup code
  });

  test('should do something specific', async ({ page }) => {
    // Arrange
    await page.goto('/path');
    
    // Act
    await page.getByRole('button').click();
    
    // Assert
    await expect(page).toHaveURL(/expected-path/);
  });
});
```

### Best Practices

- ✅ One test = one behavior
- ✅ Use descriptive test names
- ✅ Group related tests with `describe()`
- ✅ Use `beforeEach`/`afterEach` for setup/cleanup
- ✅ Keep tests independent
- ✅ Use fixtures for shared setup

## Assertions

### Explicit Assertions

```typescript
// ✅ Good - explicit and clear
await expect(page.getByText('Success')).toBeVisible();
await expect(page).toHaveURL(/dashboard/);
await expect(input).toHaveValue('expected value');

// ❌ Bad - implicit assertions
await page.getByText('Success').click(); // What if it's not there?
```

### Useful Assertions

- `toBeVisible()` - Element is visible
- `toBeHidden()` - Element is hidden
- `toHaveText()` - Text content matches
- `toHaveValue()` - Input value matches
- `toHaveURL()` - URL matches pattern
- `toHaveTitle()` - Page title matches
- `toBeEnabled()` / `toBeDisabled()` - Element state

## Page Object Model

For complex applications, use Page Objects:

```typescript
// pages/LoginPage.ts
export class LoginPage {
  constructor(private page: Page) {}

  async goto() {
    await this.page.goto('/login');
  }

  async login(email: string, password: string) {
    await this.page.getByLabel('Email').fill(email);
    await this.page.getByLabel('Password').fill(password);
    await this.page.getByRole('button', { name: 'Sign in' }).click();
  }

  async isLoggedIn() {
    return await this.page.getByText('Welcome').isVisible();
  }
}

// test
test('login', async ({ page }) => {
  const loginPage = new LoginPage(page);
  await loginPage.goto();
  await loginPage.login('user@example.com', 'password');
  await expect(loginPage.isLoggedIn()).toBeTruthy();
});
```

## Test Data Management

### Use Fixtures

```typescript
// fixtures.ts
import { test as base } from '@playwright/test';

type TestFixtures = {
  testUser: { email: string; password: string };
};

export const test = base.extend<TestFixtures>({
  testUser: async ({}, use) => {
    const user = {
      email: `test-${Date.now()}@example.com`,
      password: 'TestPassword123!'
    };
    await use(user);
    // Cleanup if needed
  },
});
```

### Generate Unique Data

```typescript
const uniqueEmail = `user-${Date.now()}@example.com`;
const uniqueId = `test-${Math.random().toString(36).substr(2, 9)}`;
```

## Error Handling

### Handle Expected Errors

```typescript
test('should show error for invalid login', async ({ page }) => {
  await page.goto('/login');
  await page.getByLabel('Email').fill('invalid@example.com');
  await page.getByLabel('Password').fill('wrong');
  await page.getByRole('button', { name: 'Sign in' }).click();
  
  await expect(page.getByText('Invalid credentials')).toBeVisible();
});
```

### Mark Slow Tests

```typescript
test('slow operation', async ({ page }) => {
  test.slow(); // Marks test as slow (3x timeout)
  // ... test code
});
```

### Skip/Fixme Tests

```typescript
test.skip('feature not implemented', async ({ page }) => {
  // Test will be skipped
});

test.fixme('flaky test', async ({ page }) => {
  // Test will be skipped and marked as needing fix
});
```

## Authentication

### Reuse Authentication State

```typescript
// global-setup.ts
import { chromium } from '@playwright/test';

async function globalSetup() {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('https://example.com/login');
  await page.getByLabel('Email').fill('test@example.com');
  await page.getByLabel('Password').fill('password');
  await page.getByRole('button', { name: 'Sign in' }).click();
  await page.context().storageState({ path: 'auth.json' });
  await browser.close();
}

// playwright.config.ts
use: {
  storageState: 'auth.json',
}
```

## Debugging

### Debug Mode

```typescript
test('debug test', async ({ page }) => {
  await page.pause(); // Opens Playwright Inspector
  // ... test code
});
```

### Screenshots and Videos

```typescript
// playwright.config.ts
use: {
  screenshot: 'only-on-failure',
  video: 'retain-on-failure',
  trace: 'on-first-retry',
}
```

## Performance

### Parallel Execution

```typescript
// playwright.config.ts
workers: process.env.CI ? 2 : 4,
fullyParallel: true,
```

### Test Timeouts

```typescript
// Global timeout
test.setTimeout(60000); // 60 seconds

// Per-test timeout
test('slow test', async ({ page }) => {
  test.setTimeout(120000); // 2 minutes
});
```

## Common Pitfalls

1. **Flaky Tests**
   - Don't use arbitrary waits
   - Wait for specific conditions
   - Use proper selectors

2. **Test Isolation**
   - Don't share state between tests
   - Clean up test data
   - Use unique identifiers

3. **Over-testing**
   - Don't test implementation details
   - Test user-facing behavior
   - Focus on critical paths

4. **Maintenance**
   - Keep tests simple
   - Extract reusable code
   - Update tests when UI changes
