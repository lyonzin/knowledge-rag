package main

import (
	"fmt"
	"net/http"
)

type Config struct {
	Port int
}

func main() {
	fmt.Println("hello")
}

func handler(w http.ResponseWriter, r *http.Request) {}
