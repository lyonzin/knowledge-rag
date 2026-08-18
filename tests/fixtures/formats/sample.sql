CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email TEXT NOT NULL
);

SELECT id, email FROM users WHERE id = 1;
INSERT INTO users (email) VALUES ('a@b.com');
