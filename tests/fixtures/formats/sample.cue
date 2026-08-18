package config

#Server: {
	host: string
	port: int & >0 & <65536
}

server: #Server & {
	host: "localhost"
	port: 8080
}
