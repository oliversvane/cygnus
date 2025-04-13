use std::io::{self, Write};
use std::process::Command;

fn clear_terminal() {
    if cfg!(target_os = "windows") {
        Command::new("cmd").args(&["/C", "cls"]).status().unwrap();
    } else {
        Command::new("clear").status().unwrap();
    }
}

fn option1() {
    println!("Option1");
    main();
}

fn option2() {
    println!("Option2");
    main();
}

fn option3() {
    println!("Option3");
    main();
}

fn quit() {
    println!("Quitting")
}

fn invalid_input() {
    println!("Invalid input");
    main();
}

fn main() {
    println!("1. Option 1");
    println!("2. Option 2");
    println!("3. Option 3");
    print!("Choose (1-3): ");
    io::stdout().flush().unwrap();

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();

    clear_terminal();

    match input.trim() {
        "1" => option1(),
        "2" => option2(),
        "3" => option3(),
        "q" => quit(),
        _ => invalid_input(),
    }
}
