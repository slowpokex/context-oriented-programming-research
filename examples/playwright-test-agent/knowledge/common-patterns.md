# Common Playwright Test Patterns

This document provides common patterns and examples for Playwright tests.

## Form Testing

### Basic Form Fill and Submit

```typescript
test('should submit form', async ({ page }) => {
  await page.goto('/contact');
  await page.getByLabel('Name').fill('John Doe');
  await page.getByLabel('Email').fill('john@example.com');
  await page.getByLabel('Message').fill('Test message');
  await page.getByRole('button', { name: 'Submit' }).click();
  await expect(page.getByText('Thank you')).toBeVisible();
});
```

### Form Validation

```typescript
test('should show validation errors', async ({ page }) => {
  await page.goto('/contact');
  await page.getByRole('button', { name: 'Submit' }).click();
  await expect(page.getByText('Name is required')).toBeVisible();
  await expect(page.getByText('Email is required')).toBeVisible();
});
```

## Navigation Testing

### Basic Navigation

```typescript
test('should navigate between pages', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('link', { name: 'About' }).click();
  await expect(page).toHaveURL(/about/);
  await expect(page.getByRole('heading', { name: 'About Us' })).toBeVisible();
});
```

### Navigation with Wait

```typescript
test('should navigate and wait for content', async ({ page }) => {
  await page.goto('/products');
  await page.getByRole('link', { name: 'Product Details' }).first().click();
  await page.waitForLoadState('networkidle');
  await expect(page.getByRole('heading', { name: /Product/ })).toBeVisible();
});
```

## Authentication Patterns

### Login Flow

```typescript
test('should log in successfully', async ({ page }) => {
  await page.goto('/login');
  await page.getByLabel('Email').fill('user@example.com');
  await page.getByLabel('Password').fill('password123');
  await page.getByRole('button', { name: 'Sign in' }).click();
  
  // Wait for navigation
  await page.waitForURL(/dashboard/);
  await expect(page.getByText('Welcome')).toBeVisible();
});
```

### Login with API Wait

```typescript
test('should log in and wait for API', async ({ page }) => {
  await page.goto('/login');
  
  // Wait for login API call
  const responsePromise = page.waitForResponse(
    response => response.url().includes('/api/login') && response.status() === 200
  );
  
  await page.getByLabel('Email').fill('user@example.com');
  await page.getByLabel('Password').fill('password123');
  await page.getByRole('button', { name: 'Sign in' }).click();
  
  await responsePromise;
  await expect(page).toHaveURL(/dashboard/);
});
```

## Dynamic Content

### Waiting for Dynamic Elements

```typescript
test('should wait for dynamic content', async ({ page }) => {
  await page.goto('/dashboard');
  
  // Wait for API call to complete
  await page.waitForResponse(response => 
    response.url().includes('/api/data')
  );
  
  // Wait for content to appear
  await expect(page.getByTestId('data-table')).toBeVisible();
  await expect(page.getByTestId('data-table').getByRole('row')).toHaveCount(10);
});
```

### Handling Loading States

```typescript
test('should handle loading state', async ({ page }) => {
  await page.goto('/products');
  
  // Wait for loading to finish
  await page.waitForSelector('[data-testid="loading"]', { state: 'hidden' });
  
  // Verify content loaded
  await expect(page.getByTestId('product-list')).toBeVisible();
});
```

## File Upload

### Basic File Upload

```typescript
test('should upload file', async ({ page }) => {
  await page.goto('/upload');
  
  const fileInput = page.getByLabel('Choose file');
  await fileInput.setInputFiles('path/to/file.pdf');
  
  await page.getByRole('button', { name: 'Upload' }).click();
  await expect(page.getByText('Upload successful')).toBeVisible();
});
```

## Dropdowns and Selects

### Select from Dropdown

```typescript
test('should select from dropdown', async ({ page }) => {
  await page.goto('/settings');
  await page.getByLabel('Country').selectOption('United States');
  await expect(page.getByLabel('Country')).toHaveValue('us');
});
```

## Modals and Dialogs

### Open and Close Modal

```typescript
test('should open and close modal', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: 'Open Modal' }).click();
  
  await expect(page.getByRole('dialog')).toBeVisible();
  await expect(page.getByRole('dialog').getByText('Modal Title')).toBeVisible();
  
  await page.getByRole('button', { name: 'Close' }).click();
  await expect(page.getByRole('dialog')).toBeHidden();
});
```

## Tables and Lists

### Interact with Table

```typescript
test('should interact with table', async ({ page }) => {
  await page.goto('/users');
  
  // Wait for table to load
  await expect(page.getByRole('table')).toBeVisible();
  
  // Click first row
  await page.getByRole('row').nth(1).click();
  
  // Verify details page
  await expect(page).toHaveURL(/users\/\d+/);
});
```

## API Mocking

### Mock API Response

```typescript
test('should mock API response', async ({ page }) => {
  // Mock API before navigation
  await page.route('**/api/users', route => {
    route.fulfill({
      status: 200,
      body: JSON.stringify([{ id: 1, name: 'Test User' }])
    });
  });
  
  await page.goto('/users');
  await expect(page.getByText('Test User')).toBeVisible();
});
```

## Screenshot Comparison

### Take Screenshot

```typescript
test('should match screenshot', async ({ page }) => {
  await page.goto('/dashboard');
  await page.waitForLoadState('networkidle');
  await expect(page).toHaveScreenshot('dashboard.png');
});
```

## Multiple Tabs/Windows

### Handle New Tab

```typescript
test('should handle new tab', async ({ context, page }) => {
  await page.goto('/');
  
  const [newPage] = await Promise.all([
    context.waitForEvent('page'),
    page.getByRole('link', { name: 'Open in new tab' }).click()
  ]);
  
  await newPage.waitForLoadState();
  await expect(newPage).toHaveURL(/expected-url/);
  await newPage.close();
});
```

## Keyboard Navigation

### Keyboard Interactions

```typescript
test('should navigate with keyboard', async ({ page }) => {
  await page.goto('/');
  await page.keyboard.press('Tab'); // Focus first element
  await page.keyboard.press('Enter'); // Activate
  await expect(page).toHaveURL(/expected/);
});
```

## Drag and Drop

### Drag and Drop Element

```typescript
test('should drag and drop', async ({ page }) => {
  await page.goto('/board');
  
  const source = page.getByTestId('item-1');
  const target = page.getByTestId('drop-zone');
  
  await source.dragTo(target);
  await expect(target.getByTestId('item-1')).toBeVisible();
});
```

## Date/Time Inputs

### Fill Date Input

```typescript
test('should fill date input', async ({ page }) => {
  await page.goto('/form');
  await page.getByLabel('Date').fill('2024-01-15');
  await expect(page.getByLabel('Date')).toHaveValue('2024-01-15');
});
```

## Checkboxes and Radio Buttons

### Toggle Checkbox

```typescript
test('should toggle checkbox', async ({ page }) => {
  await page.goto('/settings');
  const checkbox = page.getByLabel('Enable notifications');
  
  await checkbox.check();
  await expect(checkbox).toBeChecked();
  
  await checkbox.uncheck();
  await expect(checkbox).not.toBeChecked();
});
```

## Error Handling

### Test Error States

```typescript
test('should handle error state', async ({ page }) => {
  // Mock error response
  await page.route('**/api/data', route => {
    route.fulfill({ status: 500, body: 'Internal Server Error' });
  });
  
  await page.goto('/dashboard');
  await expect(page.getByText('Something went wrong')).toBeVisible();
});
```

## Retry Logic

### Retry on Failure

```typescript
test('should retry on flaky operation', async ({ page }) => {
  await page.goto('/');
  
  // Retry logic
  let success = false;
  for (let i = 0; i < 3; i++) {
    try {
      await page.getByRole('button', { name: 'Submit' }).click();
      await expect(page.getByText('Success')).toBeVisible({ timeout: 5000 });
      success = true;
      break;
    } catch (e) {
      if (i === 2) throw e;
      await page.waitForTimeout(1000);
    }
  }
  
  expect(success).toBe(true);
});
```
