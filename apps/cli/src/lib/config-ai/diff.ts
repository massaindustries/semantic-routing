import chalk from 'chalk';

/**
 * Tiny line-based diff. Adequate for showing YAML config patches.
 * Uses LCS to find common subsequence then prints +/-/= prefixes.
 */
export function lineDiff(a: string, b: string): string[] {
  const A = a.split('\n');
  const B = b.split('\n');
  const m = A.length, n = B.length;
  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }
  const out: string[] = [];
  let i = 0, j = 0;
  while (i < m && j < n) {
    if (A[i] === B[j]) { out.push('  ' + A[i]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push(chalk.red('- ' + A[i])); i++; }
    else { out.push(chalk.green('+ ' + B[j])); j++; }
  }
  while (i < m) { out.push(chalk.red('- ' + A[i++])); }
  while (j < n) { out.push(chalk.green('+ ' + B[j++])); }
  return out;
}

export function summarizeDiff(a: string, b: string): { added: number; removed: number } {
  const A = a.split('\n');
  const B = b.split('\n');
  const setA = new Set(A);
  const setB = new Set(B);
  let added = 0, removed = 0;
  for (const x of B) if (!setA.has(x)) added++;
  for (const x of A) if (!setB.has(x)) removed++;
  return { added, removed };
}
