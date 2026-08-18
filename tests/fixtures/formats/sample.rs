use std::collections::HashMap;

pub struct Config {
    port: u16,
}

trait Handler {
    fn handle(&self) -> bool;
}

pub fn main() {
    println!("hello");
}
