#!/usr/bin/env node
import { execute } from '@oclif/core';
process.env.NODE_ENV = 'development';
await execute({ development: true, dir: import.meta.url });
