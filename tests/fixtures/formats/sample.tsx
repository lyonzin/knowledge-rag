// Format smoke fixture for the React TSX parser.
import React, { useState } from "react";

interface CounterProps {
    initial: number;
}

export function Counter({ initial }: CounterProps): JSX.Element {
    const [count, setCount] = useState<number>(initial);
    return <span>{count}</span>;
}
