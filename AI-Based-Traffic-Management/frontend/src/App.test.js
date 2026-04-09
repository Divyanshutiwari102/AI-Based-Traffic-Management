import { render, screen } from '@testing-library/react';
import App from './App';

jest.mock('axios', () => ({
  post: jest.fn(),
  get: jest.fn(),
}));

test('renders traffic management title', () => {
  render(<App />);
  const titleElement = screen.getByText(/AI Based Traffic Management/i);
  expect(titleElement).toBeInTheDocument();
});
