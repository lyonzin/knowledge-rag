def names: [.[] | .name];

.items[] | select(.active) | {name, id}
