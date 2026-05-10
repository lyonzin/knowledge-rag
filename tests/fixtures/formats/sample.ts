// Format smoke fixture for the TypeScript parser.
import type { Router } from "express";

interface AppConfig {
    port: number;
    host: string;
}

enum Status {
    Active,
    Inactive,
}

export function createRouter(): Router {
    return {} as Router;
}
