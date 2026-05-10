// Format smoke fixture for the JavaScript parser.
import { useState } from "react";

export function fetchData(url) {
    return fetch(url).then((r) => r.json());
}

export class DataService {
    constructor(endpoint) {
        this.endpoint = endpoint;
    }
}
