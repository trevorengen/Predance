from flask_app.config.mysqlcontroller import connectToMySQL
import re
from flask import flash

EMAIL_CHECK = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
CAPS = re.compile(r'[A-Z]')
NUMS = re.compile(r'[0-9]')
DB = 'users_schema'

class User:
    def __init__(self, data):
        self.id = data['id']
        self.first_name = data['first_name']
        self.last_name = data['last_name']
        self.email = data['email']
        self.password = data['password']
        self.created_at = data['created_at']
        self.updated_at = data['updated_at']

    @staticmethod
    def validate_registration(input):
        is_valid = True
        if len(input['first_name']) < 3 or len(input['last_name']) < 3:
            flash('First and last name must be at least 3 characters.')
            is_valid = False
        if EMAIL_CHECK.match(input['email']):
            flash('Invalid email address.')
            is_valid = False
        if len(input['password']) < 6:
            flash('Password must be at least 6 characters.')
            is_valid = False
        elif not CAPS.search(input['password']) or not NUMS.search(input['password']):
            flash('Password must contain at least one capital letter AND one number.')
            is_valid = False
        elif input['password'] != input['confirm-password']:
            flash('Passwords don\'t match.')
            is_valid = False
        return is_valid

    @classmethod
    def save(cls, data):
        query = 'INSERT INTO users (first_name, last_name, email, password) '
        query += 'VALUES (%(first_name)s, %(last_name)s, %(email)s, %(password)s);'
        user_id = connectToMySQL(DB).query_db(query, data)
        return user_id

    @classmethod
    def update(cls, data, option='p'):
        addComma = False
        query = 'UPDATE users SET '
        if kwargs == 'p':
            query += 'password = %(password)s '
            addComma = True
        if kwargs == 'e':
            if addComma:
                query += ', '
            query += 'email = %(email)s '
            addComma = True
        if kwargs == 'n':
            if addComma:
                query += ', '
            query += 'first_name = %(first_name)s, last_name = %(last_name)s '
        query += 'WHERE id == %(id)s;'
        # I'm not sure what this returns with an update, look into this.
        # user_id is just a place hold until I'm sure of its return value.
        result = connectToMySQL(DB).query_db(query, data)
        return result

    @classmethod
    def get_user(cls, data):
        query = 'SELECT * FROM users WHERE id = %(id)s'
        results = connectToMySQL(DB).query_db(query, data)
        user = cls(results[0])
        return user

    @classmethod
    def delete_user(cls, data):
        query = 'DELETE FROM users WHERE id = %(id)s AND password = %(password)s;'
        did_delete = connectToMySQL(DB).query_db(query, data)
        return did_delete 

