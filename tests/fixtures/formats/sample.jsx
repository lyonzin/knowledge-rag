// Format smoke fixture for the React JSX parser.
import React, { useState } from "react";

export function Counter({ initial }) {
    const [count, setCount] = useState(initial);
    return (
        <button onClick={() => setCount(count + 1)}>
            Count: {count}
        </button>
    );
}
