package com.example.auth

import java.time.Instant

data class Session(val token: String)

class AuthService {
    fun validate(token: String): Boolean {
        return token.isNotEmpty()
    }
}

private fun helper(): Instant = Instant.now()
