use crate::error::EntroGdError;

pub trait Filter {
    type Input;
    type Output;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError>;
}

pub struct Chain<A, B> {
    previous: A,
    next: B,
}

impl<A, B> Filter for Chain<A, B>
where
    A: Filter,
    B: Filter<Input = A::Output>,
{
    type Input = A::Input;
    type Output = B::Output;

    fn process(&self, input: Self::Input) -> Result<Self::Output, EntroGdError> {
        let intermediate = self.previous.process(input)?;
        self.next.process(intermediate)
    }
}

pub trait FilterExt: Filter + Sized {
    fn then<Next>(self, next: Next) -> Chain<Self, Next>
    where
        Next: Filter<Input = Self::Output>,
    {
        Chain {
            previous: self,
            next,
        }
    }
}

impl<F: Filter> FilterExt for F {}
